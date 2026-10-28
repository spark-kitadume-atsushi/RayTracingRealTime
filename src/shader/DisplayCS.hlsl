// shader/DisplayCS.hlsl
Texture2D<float4> AccumTexture              : register(t0);
Texture2D<float4> HistoryTexture            : register(t1);
Texture2D<float4> CurrentWorldPosTexture    : register(t2); // 現フレームの座標
Texture2D<float4> PreWorldPosTexture        : register(t3); // 前フレームの座標

RWTexture2D<float4> BackBuffer : register(u0);
RWTexture2D<float4> HistoryUAV : register(u1);

SamplerState LinearSampler : register(s0);

cbuffer DisplayBuffer : register(b0)
{
    int TotalSamples;
    int IsCameraMoving;
    float2 Pad;
    row_major float4x4 ViewProjPrev;  // 前フレームのView-Pro行列
}

// ACES Filmic Tone Mapping(トーンマッピング:白飛びを滑らかに階調化)
float3 aces_tone_map(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    BackBuffer.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;
    
    // 現在のレイの色（蓄積平均）
    float3 current_color = AccumTexture[id.xy].rgb / max(1.0f, (float)TotalSamples);
    float3 final_color = current_color;

    if (IsCameraMoving == 1)
    {
        // ---------------------------------------------------------------------
        // 【移動中】時間的再投影 (Temporal Reprojection)
        // ---------------------------------------------------------------------
        float4 gbuffer = CurrentWorldPosTexture[id.xy];
        float3 current_pos = gbuffer.xyz;
        float is_specular = gbuffer.w;      // 1.0:鏡/ガラス, 0.0:床/ディフューズ球
        
        // ワールド座標を前フレームのプロジェクション空間へ変換
        float4 prev_clip = mul(float4(current_pos, 1.0f), ViewProjPrev);

        // 画面外や深度異常（空など）の判定
        if (prev_clip.w > 0.0f)
        {
            prev_clip.xyz /= prev_clip.w;   // NDC空間[-1,1]
            
            // UV空間[0,1]への変換
            float2 prev_uv;
            prev_uv.x = prev_clip.x * 0.5f + 0.5f;
            prev_uv.y = -prev_clip.y * 0.5f + 0.5f;
            
            // 画面外へはみ出た場合、端の有効ピクセルをクランプしてサンプリング（黒帯を完全消去）
            float2 clamped_uv = clamp(prev_uv, float2(0.001f, 0.001f), float2(0.999f, 0.999f));
            float3 history_color = HistoryTexture.SampleLevel(LinearSampler, clamped_uv, 0).rgb;

            // 画面内に収まっている場合のみ過去の色を取得
            if (prev_uv.x >= 0.0f && prev_uv.x <= 1.0f && prev_uv.y >= 0.0f && prev_uv.y <= 1.0f)
            {
                // 過去フレームの色を取得
                float3 history_color = HistoryTexture.SampleLevel(LinearSampler, prev_uv, 0).rgb;

                // ----------------------------------------------------
                // 幾何距離による遮蔽判定
                // 再投影先のピクセルが持っていた前フレームのワールド座標を取得
                // 前フレーム座標との幾何遮蔽判定（角での補間崩れを防ぐためLoadで最近傍取得）
                // ----------------------------------------------------
                int2 prev_pixel = clamp(int2(prev_uv * float2(width, height)), int2(0, 0), int2(width - 1, height - 1));
                float3 prev_history_pos = PreWorldPosTexture.Load(int3(prev_pixel, 0)).xyz;
                
                // 物体の裏から別の物体が現れたか（距離差が大きいか）を判定
                float pos_diff = length(current_pos - prev_history_pos);
                float disocclude_thresh = (is_specular > 0.5f) ? 1.0f : 0.5f;
                bool is_disoccluded = (pos_diff > disocclude_thresh);
                if (is_disoccluded)
                {
                    // 遮蔽された箇所は現在色を多めにブレンド
                    final_color = lerp(history_color, current_color, 0.5f);
                }
                else
                {
                    // 同一面上：過去88%：現在12%で滑らかに維持
                    // 鏡面は残像を防ぐために過去の重みを下げて追従
                    float blend_weight = (is_specular > 0.5f) ? 0.25f : 0.12f;
                    final_color = lerp(history_color, current_color, blend_weight);
                }
            }
            else
            {
                // 画面外から新たに入ってきた境界：前フレームの端の色（明るい色）と現在色を滑らかに合成
                final_color = lerp(history_color, current_color, 0.3f);
            }
        }
    }
    else
    {
        // ---------------------------------------------------------------------
        // 【静止中】スムーズ・ハンドオフ (クロスフェード移行)
        // 止まった瞬間もサンプルが溜まるまでは前フレームの綺麗な履歴を活用する
        // ---------------------------------------------------------------------
        const int HANDOFF_SAMPLES = 32; // 32サンプルに達するまでクロスフェード
        if (TotalSamples < HANDOFF_SAMPLES)
        {
            // カメラが止まっているため、前フレームの同じピクセル位置から色を取得
            float3 history_color = HistoryTexture[id.xy].rgb;
            
            // サンプル数が増えるにつれて現在のフレームへの比率を徐々に100%へ引き上げる
            float blend_to_current = (float) TotalSamples / (float) HANDOFF_SAMPLES;
            final_color = lerp(history_color, current_color, blend_to_current);
        }
        else
        {
            // 32サンプル異常たまったら完全な静止画蓄積へ移行
            final_color = current_color;
        }
    }
    
    // 履歴バッファを更新
    HistoryUAV[id.xy] = float4(final_color, 1.0f);
    
    // トーンマッピングとガンマ補正
    float3 mapped = aces_tone_map(final_color);
    float3 srgb = pow(mapped, 1.0f / 2.2f);
    BackBuffer[id.xy] = float4(srgb, 1.0f);
}
