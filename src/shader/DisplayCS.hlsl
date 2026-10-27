// shader/DisplayCS.hlsl
Texture2D<float4> AccumTexture : register(t0);
RWTexture2D<float4> BackBuffer : register(u0);

cbuffer DisplayBuffer : register(b0)
{
    int TotalSamples;
    float3 Pad;
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
    
    // 蓄積された光をサンプル数で割って平均化する
    float3 linear_color = AccumTexture[id.xy].rgb / max(1.0f, (float)TotalSamples);

    // 1.トーンマッピング
    float3 mapped = aces_tone_map(linear_color);
    
    // 2.ガンマ補正（リニア->sRGB）
    float3 srgb = pow(mapped, 1.0f / 2.2f);

    BackBuffer[id.xy] = float4(srgb, 1.0f);
}
