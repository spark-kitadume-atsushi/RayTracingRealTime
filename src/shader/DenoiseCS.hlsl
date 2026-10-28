// shader/DenoiseCS.hlsl
Texture2D<float4> ColorTexture          : register(t0); // 生カラー
Texture2D<float4> WorldPosTexture       : register(t1); // 座標(xyz) + 鏡面フラグ(w)
Texture2D<float4> NormalTexture         : register(t2); // 法線(xyz) + アルベド輝度(w)

RWTexture2D<float4> DenoisedTexture     : register(u0);

cbuffer DenoiseBuffer : register(b0)
{
    int TotalSamples;
    int IsCameraMoving;
    float2 Pad;
}

float calc_albedo(float3 alb)
{
    return dot(alb, float3(0.299f, 0.587f, 0.114f));
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    DenoisedTexture.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;
    
    // 現在の中心ピクセルの生光量
    float3 center_color = ColorTexture[id.xy].rgb;
    float center_lum = calc_albedo(center_color);
    
    float4 g_pos = WorldPosTexture[id.xy];
    float3 pos0 = g_pos.xyz;
    float spec0 = g_pos.w;      // 1.0:鏡/ガラス, 0.0:ディフューズ
    
    float4 g_norm = NormalTexture[id.xy];
    float3 norm0   = g_norm.xyz;
    float lum0 = g_norm.w;      // アルベド輝度
    
    // 鏡・ガラス球は像が複雑なためフィルタ半径を絞る
    int radius = (spec0 > 0.5f) ? 1 : 2;    // ディフューズ面は5x5、鏡面は3x3

    float3 accum_color = float3(0.0f, 0.0f, 0.0f);
    float accum_weight = 0.0f;
    
    // フィルタパラメータ
    const float sigma_spatial = 1.4f;
    const float sigma_plane = 0.04f;    // 平面からのずれ許容値(4cm)
    
    [unroll]
    for (int dy = -2; dy <= 2; ++dy)
    {
        [unroll]
        for (int dx = -2; dx <= 2; ++dx)
        {
            if (abs(dx) > radius || abs(dy) > radius)
                continue;
            
            int2 sample_coord = clamp(int2(id.xy) + int2(dx, dy), int2(0, 0), int2(width - 1, height - 1));
            
            // クランプ済みの色を使用
            float3 c_i = ColorTexture[sample_coord].rgb;
            float4 p_i = WorldPosTexture[sample_coord];
            float4 n_i = NormalTexture[sample_coord];

            // 1.空間距離重み（中心に近いほど強い）
            float dist_sq = float(dx * dx + dy * dy);
            float w_spatial = exp(-dist_sq / (2.0f * sigma_spatial * sigma_spatial));
            
            // 2.法線内積重み（面の向きが異なる場合は遮断）
            float normal_dot = max(0.0f, dot(norm0, n_i.xyz));
            float w_normal = pow(normal_dot, 32.0f);
            
            // 3.平面距離重み（接平面から浮いている場合は遮断）
            float plane_dist = abs(dot(norm0, p_i.xyz - pos0));
            float w_plane = exp(-plane_dist / sigma_plane);

            // 4.アルベド輝度差重み (チェッカー境界を保護しつつ滑らかに減衰)
            // 二値判定(0/1)を撤廃し、滑らかな指数減衰に統一
            float lum_diff = abs(lum0 - n_i.w);
            float w_albedo = exp(-lum_diff / 0.15f);

            // 5. 鏡面・非鏡面の一致度重み
            float spec_diff = abs(spec0 - p_i.w);
            float w_spec = exp(-spec_diff / 0.2f);
            
            // 6.輝度差に応じた相対重み
            // 孤立した強烈な交点が周囲に四角いブロックとして張り付くのを禁止
            float sample_lum = calc_albedo(c_i);
            float lum_ratio = abs(center_lum - sample_lum) / max(0.1f, max(center_lum, sample_lum));
            float w_lum = exp(-lum_ratio * 0.5f);

            // 総合重み
            float weight = w_spatial * w_normal * w_plane * w_albedo * w_spec * w_lum;
            
            accum_color += c_i * weight;
            accum_weight += weight;
        }
    }
    
    float3 filtered = (accum_weight > 1e-4f) ? (accum_color / accum_weight) : center_color;
    DenoisedTexture[id.xy] = float4(filtered, 1.0f);
}
