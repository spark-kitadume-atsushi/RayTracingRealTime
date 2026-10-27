// shader/DisplayCS.hlsl
Texture2D<float4> AccumTexture : register(t0);
Texture2D<float4> HistoryTexture : register(t1);
Texture2D<float4> WorldPosTexture : register(t2);

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
        float3 world_pos = WorldPosTexture[id.xy].xyz;
        
        // ワールド座標を前フレームのプロジェクション空間へ変換
        float4 prev_clip = mul(float4(world_pos, 1.0f), ViewProjPrev);

        // 画面外や深度異常（空など）の判定
        if (prev_clip.w > 0.0f)
        {
            prev_clip.xyz /= prev_clip.w;   // NDC空間[-1,1]
            
            // UV空間[0,1]への変換
            float2 prev_uv;
            prev_uv.x = prev_clip.x * 0.5f + 0.5f;
            prev_uv.y = -prev_clip.y * 0.5f + 0.5f;
            
            // 画面内に収まっている場合のみ過去の色を取得
            if (prev_uv.x >= 0.0f && prev_uv.x <= 1.0f && prev_uv.y >= 0.0f && prev_uv.y <= 1.0f)
            {
                float3 history_color = HistoryTexture.SampleLevel(LinearSampler, prev_uv, 0).rgb;

                // ブレンド率（現在15%：過去85%で滑らかに繋ぐ）
                final_color = lerp(history_color, current_color, 0.15f);
            }
        }

    }
    
    // 履歴バッファを更新
    HistoryUAV[id.xy] = float4(final_color, 1.0f);
    
    // トーンマッピングとガンマ補正
    float3 mapped = aces_tone_map(final_color);
    float3 srgb = pow(mapped, 1.0f / 2.2f);
    BackBuffer[id.xy] = float4(srgb, 1.0f);
}
