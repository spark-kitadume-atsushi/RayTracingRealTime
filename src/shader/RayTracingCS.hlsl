// RayTracingCS.hlsl

static const float PI     = 3.14159265358979323846f;
static const float TWO_PI = 6.28318530717958647692f;

// サンプリング対象の識別(0: Sphere, 1:Quad)
struct LightTarget
{
    int type;   // 0:Sphere, 1:Quad
    int index;  // Spheres または Quads 配列のインデックス
};

// 最大サンプリング対象数（4個程度あれば十分です）
static const int MAX_LIGHT_TARGETS = 4;

// 被写界深度パラメータを追加したカメラ定数バッファ（112bytes:16の倍数）
cbuffer CameraBuffer : register(b0)
{
    float3 CameraCenter;
    int PassIndex;          // 現在のパス番号
    float3 Pixel00Loc;
    int SamplesThisPass;
    float3 PixelDeltaU;
    int MaxDepth;
    float3 PixelDeltaV;
    float DefocusAngle;
    float3 DefocusDiskU;
    int NumLightTargets;    // サンプリング対称数
    float3 DefocusDiskV;
    float Pad1;
    float3 BackgroundColor;
    float Pad2;
    int4 LightTargetData[MAX_LIGHT_TARGETS];    // x:type, y:index
}

// マテリアル(48bytes)
struct Material
{
    float3 albedo;  // 基本色 / チェッカー1
    int type;       // 0: Lambertian, 1: Metal, 2: Dielectric
    float3 albedo2; // チェッカー2
    int text_type;  // 0:Solid 1:Checker 2:Image
    float param;    // fuzz / ref_idx
    float scale;    // checker scale
    float2 pad;
};

// 球体
struct Sphere
{
    float3 center;
    float radius;
    float3 center_vec;
    float pad;
    Material mat;
};

// 四角形プリミティブ(128bytes)
struct Quad
{
    float3 Q;       // 基準点
    float D;        // 平面パラメーター(n.Q)
    float3 u;       // 辺ベクトル 1
    float pad0;
    float3 v;       // 辺ベクトル 2
    float pad1;
    float3 normal;  // 単位法線
    float pad2;
    float3 w;       // 内外判定用補助ベクトル: n / (n.(u x v))
    float pad3;
    Material mat;
};

// 参加媒質ボリューム(96bytes)
struct Volume
{
    int shape_type; //  0:sphere 1:Box
    float density; // 媒質密度
    float2 pad0;
    float3 albedo; // 散乱色
    float pad1;
    float3 center; // Sphere 中心
    float radius; // Sphere 半径
    float3 box_min; // Box 最小点
    float rotate_y_deg; // Box Y軸回転角
    float3 box_max; // Box 最大点
    float pad2;
    float3 translation; // Box 平行移動距離
    float pad3;
};

// レイの定義
struct Ray
{
    float3 origin;
    float3 direction;
    float time;         // シャッター時間[0,1]
};

// 当たり判定の情報
struct HitRecord
{
    float t;
    float3 p;
    float3 normal;
    float2 uv;
    bool front_face;
    Material mat;
};

// 散乱結果を格納する構造体
struct ScatterRecord
{
    bool is_specular; // 鏡面反射(Metal/Dielectric)かどうかのフラグ
    Ray scattered_ray; // 生成された散乱レイ
    float3 attenuation; // マテリアルのアルベド
};

// AABB（境界ボックス:32bytes）
struct Aabb
{
    float3 min_point;
    float pad0;
    float3 max_point;
    float pad1;
};

// フラットBVHノード（48bytes）
struct BvhNode
{
    Aabb bbox;
    int left_child_index;
    int right_child_index;
    int primitive_type;     // -1: Branch(内部ノード), 0: Sphere, 1: Quad
    int primitive_index;    // 各配列内のインデックス
};

// ---------------------------------------------------
// 正規直交基底 (ONB) 構造体と関数
// ---------------------------------------------------
struct ONB
{
    float3 u;
    float3 v;
    float3 w;
};

ONB build_onb(float3 n)
{
    ONB onb;
    onb.w = normalize(n);
    // wがX軸に近すぎる場合の特異点を回避
    float3 a = (abs(onb.w.x) > 0.9f) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    onb.v = normalize(cross(onb.w, a));
    onb.u = cross(onb.w, onb.v);
    return onb;
}

// ローカル座標からワールド空間への変換
float3 onb_local(ONB onb, float3 a)
{
    return a.x * onb.u + a.y * onb.v + a.z * onb.w;
}

// 出力先UAVの定義（UAV:u0 WorldPos:u1 Normal:u2）
RWTexture2D<float4> OutputTexture   : register(u0);
RWTexture2D<float4> WorldPosTexture : register(u1);
RWTexture2D<float4> NormalTexture   : register(u2); // 法線(xyz)とアルベド輝度(w)
// C++から転送された球体バッファ(t0)とBVHノードバッファ(t1)
StructuredBuffer<Sphere> Spheres : register(t0);
StructuredBuffer<BvhNode> BvhNodes : register(t1);

// 画像テクスチャ（t2）とバイリニアサンプラー（s0）
Texture2D<float4> ImageTexture : register(t2);
SamplerState LinearSampler : register(s0);

// 四角形バッファ
StructuredBuffer<Quad> Quads : register(t3);

// Volumeバッファ
StructuredBuffer<Volume> Volumes : register(t4);
//--------------------------------------------
// 高速疑似乱数（PCG Hash）
//--------------------------------------------
uint pcg_hash(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint world = (state >> ((state >> 28u) + 4u) ^ state) * 277803737u;
    return (world >> 22u) ^ world;
}

float rand_float(inout uint state)
{
    return (pcg_hash(state) & 0x00FFFFFFu) / 16777216.0f;
}

float3 random_uint_vector(inout uint state)
{
    // 球面座標系による一様サンプリング
    float z = rand_float(state) * 2.0f - 1.0f;
    float a = rand_float(state) * TWO_PI;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    return float3(r * cos(a), r * sin(a), z);
}

// レンズ絞り（単位円盤）内のサンプリング
float3 random_in_unit_disk(inout uint state)
{
    // 単位円盤内の一様分布：半径rは平方根、角度 theraは0～2π
    float r = sqrt(rand_float(state));
    float theta = rand_float(state) * TWO_PI;
    
    return float3(r * cos(theta), r * sin(theta), 0.0f);
}

// 単位法線ベクトルから球面UV座標
float2 get_sphere_uv(float3 outward_normal)
{
    float theta = acos(-outward_normal.y);
    float phi = atan2(-outward_normal.z, outward_normal.x) + PI;
    return float2(phi / (TWO_PI), 1.0f - theta / PI);
}

//--------------------------------------------
// パーリンノイズ（3D勾配ノイズ）
//--------------------------------------------

// 高品質な3D整数ハッシュ(PCG3D)
uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// 格子点(int3)から決定論的な単位勾配ベクトルを生成するハッシュ
float3 hash_gradient(int3 p)
{
    uint3 u = (uint3) p;
    uint3 h = pcg3d(u);

    // [-1,1]のベクトルにして正規化
    float3 g = (float3(h) / 4294967295.0f) * 2.0f - 1.0f;
    return normalize(g);
}

// 単一オクターブの3Dパーリンノイズ[-1,1]
float perlin_noise(float3 p)
{
    int3 i = int3(floor(p));
    float3 f = frac(p);

    // 5次エルミート補間（滑らかな微笑変化:6f^5 - 15f^4 + 10f^3）
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    // 8つの格子点の寄与を計s何して三直線補間
    float n000 = dot(hash_gradient(i + int3(0, 0, 0)), f - float3(0, 0, 0));
    float n100 = dot(hash_gradient(i + int3(1, 0, 0)), f - float3(1, 0, 0));
    float n010 = dot(hash_gradient(i + int3(0, 1, 0)), f - float3(0, 1, 0));
    float n110 = dot(hash_gradient(i + int3(1, 1, 0)), f - float3(1, 1, 0));
    float n001 = dot(hash_gradient(i + int3(0, 0, 1)), f - float3(0, 0, 1));
    float n101 = dot(hash_gradient(i + int3(1, 0, 1)), f - float3(1, 0, 1));
    float n011 = dot(hash_gradient(i + int3(0, 1, 1)), f - float3(0, 1, 1));
    float n111 = dot(hash_gradient(i + int3(1, 1, 1)), f - float3(1, 1, 1));
    
    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);

    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);

    return lerp(nxy0, nxy1, u.z);
}

// タービュランス（多重低周波の絶対値加算）
float perlin_turb(float3 p, int depth)
{
    float accum = 0.0f;
    float3 temp_p = p;
    float weight = 1.0f;
    
    [unroll]
    for (int i = 0; i < 7; i++)
    {
        if (i >= depth)
            break;
        
        accum += weight * perlin_noise(temp_p);
        weight *= 0.5f;
        temp_p *= 2.0f;
    }

    return abs(accum);
}

// ---------------------------------------------------
// コサイン半球サンプリング
// ---------------------------------------------------
float3 random_cosine_direction(inout uint rng)
{
    float r1 = rand_float(rng);
    float r2 = rand_float(rng);
    
    float phi = TWO_PI * r1;
    float x = cos(phi) * sqrt(r2);
    float y = sin(phi) * sqrt(r2);
    float z = sqrt(max(0.0f, 1.0f - r2)); // Z軸方向(法線方向)の成分
    
    return float3(x, y, z);
}

// テクスチャ種別に応じた表面色を計算
float3 get_material_albedo(Material mat, float3 p, float2 uv)
{
    if (mat.text_type == 1)              // 3D Checker
    {
        float inv_scale = 1.0f / mat.scale;

        // 【修正】0境界での floor の正負反転を防ぐため、微小オフセット (0.001f) を加えて格子を安定化
        float3 shifted_p = p + 0.001f;
        int3 ip = int3(floor(inv_scale * shifted_p));
        
        // 負の数でも安全に偶奇を判定する
        int parity = (ip.x + ip.y + ip.z) & 1;
        return (parity == 0) ? mat.albedo : mat.albedo2;
    }
    else if (mat.text_type == 2)    // Image Texture
    {
        // Compute ShaderではSampleLevelを使用
        return ImageTexture.SampleLevel(LinearSampler, uv, 0).rgb;
    }
    else if (mat.text_type == 3)    // Perlin Marble Noise
    {
        // 大理石模様
        float s = 0.5f * (1.0f + sin(mat.scale * p.z + 10.0f * perlin_turb(p, 7)));
        return s * mat.albedo;
    }
    
    // 単色（Solid）
    return mat.albedo;
}

// 球との交差判定関数
// 衝突した場合は最小のt>0を返し、衝突しない場合は府の値を返す
bool hit_sphere(Sphere s, Ray r, float ray_tmin, float ray_tmax, inout HitRecord rec)
{
    // コンパイラの未初期化チェックを通すため冒頭で0クリアする
    rec = (HitRecord) 0;

    // 時間 r.timeにおける球の中心座標
    float3 current_center = s.center + r.time * s.center_vec;
    
    float3 oc = current_center - r.origin;

    // r.directionは単位ベクトルなので計算を除外
    float h = dot(r.direction, oc);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = h * h - c;

    if (discriminant < 0.0f)
    {
        return false;
    }
    
    float sqrtd = sqrt(discriminant);
    float root = (h - sqrtd);
    
    if (root <= ray_tmin || ray_tmax <= root)
    {
        root = (h + sqrtd);
        if (root <= ray_tmin || ray_tmax <= root)
            return false;
    }
    
    rec.t = root;
    rec.p = r.origin + rec.t * r.direction;
    float3 outward_normal = (rec.p - current_center) / s.radius;
    rec.front_face = dot(r.direction, outward_normal) < 0.0f;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
    rec.uv = get_sphere_uv(outward_normal);     // UV座標を格納
    rec.mat = s.mat;
    return true;
}

// 高速・低レジスタ・早期脱出型の AABB 判定
bool hit_aabb(Aabb box, Ray r, float3 inv_d, float3 neg_orig_inv_d, float ray_tmin, float ray_tmax)
{
    [unroll]
    for (int a = 0; a < 3; ++a)
    {
        // mad(m, a, b) は m * a + b を 1 クロックで実行
        float t0 = mad(box.min_point[a], inv_d[a], neg_orig_inv_d[a]);
        float t1 = mad(box.max_point[a], inv_d[a], neg_orig_inv_d[a]);

        // スカラーのハードウェア min/max（レジスタ消費1枠のみ）
        float t_min = min(t0, t1);
        float t_max = max(t0, t1);

        ray_tmin = max(t_min, ray_tmin);
        ray_tmax = min(t_max, ray_tmax);
        
        // 【最重要】外れた軸の時点で即座に計算を打ち切る
        if (ray_tmax < ray_tmin)
            return false;
    }

    return true;
}

// 四角形の衝突判定
bool hit_quad(Quad q, Ray r, float ray_tmin, float ray_tmax, inout HitRecord rec)
{
    float denom = dot(q.normal, r.direction);
    
    // レイと平面が平衡に近い場合は交差なし
    if (abs(denom) < 1e-8f)
        return false;
    
    // 平面との交差時刻t
    float t = (q.D - dot(q.normal, r.origin)) / denom;
    if (t <= ray_tmin || t >= ray_tmax)
        return false;

    // 衝突点pが四角形の内部にあるかを判定
    float3 intersection = r.origin + t * r.direction;
    float3 planar_hitpt_vector = intersection - q.Q;
    float alpha = dot(q.w, cross(planar_hitpt_vector, q.v));

    // 早期脱出
    if (alpha < 0.0f || alpha > 1.0f)
        return false;

    float beta = dot(q.w, cross(q.u, planar_hitpt_vector));
    if (beta < 0.0f || beta > 1.0f)
        return false;
    
    rec.t = t;
    rec.p = intersection;
    rec.uv = float2(alpha, beta);
    rec.front_face = dot(r.direction, q.normal) < 0.0f;
    rec.normal = rec.front_face ? q.normal : -q.normal;
    rec.mat = q.mat;
    return true;
}

// Quad上のランダム点への方向ベクトルを生成
float3 quad_sample_direction(Quad q, float3 origin, inout uint rng)
{
    float3 random_point = q.Q + rand_float(rng) * q.u + rand_float(rng) * q.v;
    return random_point - origin;
}

// レイ方向に対するQuadの立体角PDF値を計算
float quad_pdf_value(Quad q, float3 origin, float3 direction)
{
    Ray r;
    r.origin = origin;
    r.direction = direction;
    r.time = 0.0f;
    
    HitRecord rec = (HitRecord) 0;
    // わずかに押し出した位置から交差判定
    if (!hit_quad(q, r, 0.001f, 1e9f, rec))
        return 0.0f;

    // direction は正規化済みのため rec.t がそのまま距離
    float distance_squared = rec.t * rec.t;
    
    // 平行四辺形の面積 |u x v|
    float area = length(cross(q.u, q.v));
    if (area < 1e-8f)
        return 0.0f;

    // 光源面の法線とレイ方向のコサイン
    float cosine = abs(dot(direction, q.normal));
    if (cosine < 1e-8f)
        return 0.0f;

    return distance_squared / (cosine * area);
}

// 点Pから球体を見込む包含円錐無いの一様ランダム方向を生成
float3 sphere_sample_direction(Sphere s, float3 origin, inout uint rng)
{
    float3 to_center = s.center - origin;
    float dist_sq = dot(to_center, to_center);

    // 点が球の内部または表面にある場合のガード
    float r_sq = s.radius * s.radius;
    if (dist_sq <= r_sq)
        return random_uint_vector(rng);
    
    float cos_theta_max = sqrt(max(0.0f, 1.0f - r_sq / dist_sq));
    
    float r1 = rand_float(rng);
    float r2 = rand_float(rng);
    
    float z = 1.0f + r2 * (cos_theta_max - 1.0f);
    float phi = TWO_PI * r1;
    float sin_theta = sqrt(max(0.0f, 1.0f - z * z));
    float x = cos(phi) * sin_theta;
    float y = sin(phi) * sin_theta;

    ONB onb = build_onb(to_center);
    return normalize(onb_local(onb, float3(x, y, z)));
}

// 球体に対する立体核PDFの評価
float sphere_pdf_value(Sphere s, float3 origin, float3 direction)
{
    Ray r;
    r.origin = origin;
    r.direction = direction;
    r.time = 0.0f;
    
    HitRecord rec = (HitRecord) 0;
    // レイが球体に当たらなければPDFは0
    if (!hit_sphere(s, r, 0.001f, 1e9f, rec))
        return 0.0f;
    
    float3 to_center = s.center - origin;
    float dist_sq = dot(to_center, to_center);
    float r_sq = s.radius * s.radius;
    if (dist_sq < r_sq)
        return 0.0f;

    float cos_theta_max = sqrt(max(0.0f, 1.0f - r_sq / dist_sq));
    float solid_angle = TWO_PI * (1.0f - cos_theta_max);
    return 1.0f / max(1e-8f, solid_angle);
}

// ターゲット全体の中からr何ダムに1つ選んでサンプリング方向を生成
float3 sample_light_targets(float3 origin, inout uint rng)
{
    if (NumLightTargets <= 0)
        return float3(0.0f, 1.0f, 0.0f);

    int idx = min((int) (rand_float(rng) * NumLightTargets), NumLightTargets - 1);
    int type = LightTargetData[idx].x;
    int prim_idx = LightTargetData[idx].y;
    
    if (type == 0)  // Sphere
        return sphere_sample_direction(Spheres[prim_idx], origin, rng);
    else // Quad
        return normalize(quad_sample_direction(Quads[prim_idx], origin, rng));
}

// 生成された方向に対するターゲット全体の平均PDF値を計算
float light_targets_pdf_value(float3 origin, float3 direction)
{
    if (NumLightTargets <= 0)
        return 0.0f;
    
    float sum_pdf = 0.0f;
    [unroll]
    for (int i = 0; i < MAX_LIGHT_TARGETS; ++i)
    {
        if (i >= NumLightTargets)
            break;
        
        int type = LightTargetData[i].x;
        int prim_idx = LightTargetData[i].y;
        
        if (type == 0)
            sum_pdf += sphere_pdf_value(Spheres[prim_idx], origin, direction);
        else
            sum_pdf += quad_pdf_value(Quads[prim_idx], origin, direction);
    }
    return sum_pdf / (float) NumLightTargets;
}

// Volume衝突判定
bool hit_volume(Volume vol, Ray r, float ray_tmin, float ray_tmax, inout uint rng, inout HitRecord rec)
{
    float t_enter = -1e9f;
    float t_exit = 1e9f;
    
    if (vol.shape_type == 0) // Sphere
    {
        float3 oc = vol.center - r.origin;
        float a = dot(r.direction, r.direction);
        float h = dot(r.direction, oc);
        float c = dot(oc, oc) - vol.radius * vol.radius;
        float discriminant = h * h - a * c;

        if (discriminant < 0.0f)
            return false;
        
        float sqrtd = sqrt(discriminant);
        t_enter = (h - sqrtd) / a;
        t_exit = (h + sqrtd) / a;
    }
    else    // Box
    {
        // ワールド空間のレイを箱のローカル空間へ逆変換
        float3 p = r.origin - vol.translation;
        float3 d = r.direction;
        
        if (abs(vol.rotate_y_deg) > 1e-4f)
        {
            float rad = -vol.rotate_y_deg * (PI / 180.0f);
            float s = sin(rad);
            float c = cos(rad);
            p = float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
            d = float3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
        }

        [unroll]
        for (int i = 0; i < 3; ++i)
        {
            float dir_component = abs(d[i]) > 1e-8f ? d[i] : (d[i] >= 0.0f ? 1e-8f : -1e-8f);
            float invD = 1.0f / dir_component;
            float t0 = (vol.box_min[i] - p[i]) * invD;
            float t1 = (vol.box_max[i] - p[i]) * invD;
            if (invD < 0.0f)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            t_enter = max(t_enter, t0);
            t_exit = min(t_exit, t1);
            if (t_exit <= t_enter)
                return false;
        }
    }

    t_enter = max(t_enter, ray_tmin);
    t_exit = min(t_exit, ray_tmax);

    if (t_enter >= t_exit)
        return false;
        
    if (t_enter < 0.0f)
        t_enter = 0.0f;

    float distance_inside_boundary = t_exit - t_enter;
    float hit_distance = -(1.0f / vol.density) * log(max(1e-7f, rand_float(rng)));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = t_enter + hit_distance;
    rec.p = r.origin + rec.t * r.direction;
    rec.normal = float3(1.0f, 0.0f, 0.0f); // 任意のダミー法線
    rec.uv = float2(0.0f, 0.0f);
    rec.front_face = true;
    rec.mat.type = 4; // 4: Isotropic:等方
    rec.mat.albedo = vol.albedo;
    return true;
}

// BVHを平坦化した上で実行する
bool hit_world(Ray r, float ray_tmin, float ray_tmax, inout uint rng, inout HitRecord rec)
{
    HitRecord temp_rec = (HitRecord)0;  // ゼロ初期化
    bool hit_anything = false;
    float closest_so_far = ray_tmax;

    // 【追加】レイの逆数方向を探索開始時に1度だけ計算（ゼロ除算・NaN対策）
    float3 inv_d = float3(
        abs(r.direction.x) > 1e-8f ? 1.0f / r.direction.x : (r.direction.x >= 0 ? 1e8f : -1e8f),
        abs(r.direction.y) > 1e-8f ? 1.0f / r.direction.y : (r.direction.y >= 0 ? 1e8f : -1e8f),
        abs(r.direction.z) > 1e-8f ? 1.0f / r.direction.z : (r.direction.z >= 0 ? 1e8f : -1e8f)
    );
    
    // 【追加】原点と逆数方向の積を事前計算（-r.origin * inv_d）
    float3 neg_orig_inv_d = -r.origin * inv_d;
    
    // GPUローカルメモリにスタックを確保（深さ64あれば数万個のオブジェクトに対応可能）
    // スクラッチメモリのスピルを抑えるためサイズを 20 に設定
    static const int stack_size = 64;
    int stack[stack_size];
    int stack_ptr = 0;
    
    // ルートノード
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0)
    {
        int node_idx = stack[--stack_ptr];
        BvhNode node = BvhNodes[node_idx];

        // ノードのAABBに当たらなければこのノードおよびその配下はスキップ
        if (!hit_aabb(node.bbox, r, inv_d, neg_orig_inv_d, ray_tmin, closest_so_far))
            continue;

        if (node.primitive_type >= 0)
        {
            // 葉ノードチェック
            bool hit = false;
            if (node.primitive_type == 0)       // Sphere
            {
                hit = hit_sphere(Spheres[node.primitive_index], r, ray_tmin, closest_so_far, temp_rec);
            }
            else if (node.primitive_type == 1)  // Quad
            {
                hit = hit_quad(Quads[node.primitive_index], r, ray_tmin, closest_so_far, temp_rec);
            }
            else if (node.primitive_type == 2)  // Volume
            {
                hit = hit_volume(Volumes[node.primitive_index], r, ray_tmin, closest_so_far, rng, temp_rec);
            }
            
            if (hit)
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        else
        {
            // 内部ノードの場合、node.sphere_index = -1が与えられる
            // Push前に左右の子ノードのAABBを判定し、当たったものだけを積む
            int left_idx = node.left_child_index;
            int right_idx = node.right_child_index;

            bool hit_left = (left_idx >= 0) && hit_aabb(BvhNodes[left_idx].bbox, r, inv_d, neg_orig_inv_d, ray_tmin, closest_so_far);
            bool hit_right = (right_idx >= 0) && hit_aabb(BvhNodes[right_idx].bbox, r, inv_d, neg_orig_inv_d, ray_tmin, closest_so_far);
            
            // 各Pushごとに1枠以上の空きがあるか（stack_ptr < stack_size）を判定
            if (hit_right && stack_ptr < stack_size)
            {
                stack[stack_ptr++] = right_idx;
            }
            
            if (hit_left && stack_ptr < stack_size)
            {
                stack[stack_ptr++] = left_idx;
            }
        }
    }

    return hit_anything;
}

//--------------------------------------------
// マテリアルの散乱計算(Lambertian / Metal / Dielectric)
//--------------------------------------------
float reflectance(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

// マテリアルの発光色を取得
float3 emitted(HitRecord rec)
{
    if (rec.mat.type == 3)  // 3:DiffuseLight
    {
        if (rec.front_face)
            return get_material_albedo(rec.mat, rec.p, rec.uv);
    }
    return float3(0.0f, 0.0f, 0.0f);
}

// 散乱PDFの評価関数
float scattering_pdf(Material mat, HitRecord rec, Ray scattered)
{
    if (mat.type == 0) // Lambertian
    {
        float cosine = dot(rec.normal, scattered.direction);
        return cosine < 0.0f ? 0.0f : cosine / PI;
    }
    // 鏡面反射(Metal/Dielectric)はデルタ分布として扱うため0を返す
    return 0.0f;
}

// hit_world の最小交差距離 0.001f よりも確実に外へ逃がすため 0.002f に設定
static const float normal_buffer = 0.002f;

bool scatter(Ray r_in, HitRecord rec, inout uint rng, out ScatterRecord srec)
{
    static const float normal_buffer = 0.002f;
    srec = (ScatterRecord) 0; // ゼロ初期化
    
    if (rec.mat.type == 3)  // DiffuseLight
    {
        return false;
    }
    else if (rec.mat.type == 0) // Lambertian
    {
        srec.is_specular = false;
        srec.attenuation = get_material_albedo(rec.mat, rec.p, rec.uv);
        
        // ONBを用いてコサイン半球分布に基づく散乱方向を生成
        float3 outward_dir = rec.front_face ? rec.normal : -rec.normal;
        ONB onb = build_onb(outward_dir);
        float3 local_dir = random_cosine_direction(rng);
        float3 scatter_dir = onb_local(onb, local_dir);
        
        srec.scattered_ray.origin = rec.p + outward_dir * normal_buffer;
        srec.scattered_ray.direction = normalize(scatter_dir);
        srec.scattered_ray.time = r_in.time;
        return true;
    }
    else if (rec.mat.type == 1) // Metal
    {
        srec.is_specular = true;
        srec.attenuation = get_material_albedo(rec.mat, rec.p, rec.uv);
        
        float3 reflected = reflect(r_in.direction, rec.normal);
        float3 scatter_direction = reflected + rec.mat.param * random_uint_vector(rng);
        
        srec.scattered_ray.origin = rec.p + rec.normal * normal_buffer;
        srec.scattered_ray.direction = normalize(scatter_direction);
        srec.scattered_ray.time = r_in.time;
        return dot(srec.scattered_ray.direction, rec.normal) > 0.0f;
    }
    else if (rec.mat.type == 2) // Dielectric
    {
        srec.is_specular = true;
        srec.attenuation = float3(1.0f, 1.0f, 1.0f);
        float ri = rec.front_face ? (1.0f / rec.mat.param) : rec.mat.param;
        
        float cos_theta = min(dot(-r_in.direction, rec.normal), 1.0f);
        float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));

        bool cannot_refract = (ri * sin_theta) > 1.0f;
        float3 direction;
        
        bool is_reflection = cannot_refract || reflectance(cos_theta, ri) > rand_float(rng);
        if (is_reflection)
            direction = reflect(r_in.direction, rec.normal);
        else
            direction = refract(r_in.direction, rec.normal, ri);
        
        // 自己交差を絶つためのオフセット
        float3 offset_dir = is_reflection ? rec.normal : -rec.normal;
        
        srec.scattered_ray.origin = rec.p + offset_dir * normal_buffer;
        srec.scattered_ray.direction = normalize(direction);
        srec.scattered_ray.time = r_in.time;
        return true;
    }
    else //if (rec.mat.type == 4) // Isotropic (Constant Medium)
    {
        srec.is_specular = true;
        srec.scattered_ray.origin = rec.p;
        srec.scattered_ray.direction = random_uint_vector(rng); // 全方位一様散乱
        srec.scattered_ray.time = r_in.time;
        srec.attenuation = rec.mat.albedo;
        return true;
    }
}

// エントリポイント関数。1スレッドグループあたり（16,16,1）スレッドで実行
[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    // 出力テクスチャの解像度を取得
    uint width, height;
    OutputTexture.GetDimensions(width, height);
    
    // テクスチャの範囲外スレッドは除外
    if (id.x >= width || id.y >= height)
        return;

    // 3Dハッシュ関数（pcg3d）を通して空間相関を完全に破壊する
    uint3 seed_vec = uint3(id.xy, (uint)PassIndex);
    uint3 hashed = pcg3d(seed_vec);
    
    // 3成分を結合して一様ホワイトノイズシードを作成
    uint rng = hashed.x ^ hashed.y ^ hashed.z;
    
    // ウォームアップ（初期状態の残留相関を1ステップ破棄）
    pcg_hash(rng);

    float3 pass_color = float3(0.0f, 0.0f, 0.0f);
    // 初期値は原点(0,0,0)ではなく遠方の仮想座標にして床との誤認を防止
    float3 first_hit_pos = CameraCenter + 1000.0f * (Pixel00Loc - CameraCenter);
    float3 first_hit_normal = float3(0.0f, 1.0f, 0.0f);
    float first_hit_albedo_lum = 0.5f;
    int is_first_hit_specular = 0;  // 鏡面フラグ
    
    // マルチサンプリング（アンチエイリアス）ループ
    for (int s = 0; s < SamplesThisPass; ++s)
    {
        // サンプル事に乱数状態を強く拡散
        uint sample_seed = rng ^ (uint) s * 0x9E3779B9u;
        rng = pcg_hash(sample_seed);
        
        // ピクセル平面上のサンプリング店
        float rx = rand_float(rng) - 0.5f;
        float ry = rand_float(rng) - 0.5f;
        float3 pixel_sample = Pixel00Loc + ((((float) id.x + rx) * PixelDeltaU) + (((float) id.y + ry) * PixelDeltaV));

        // 被写界深度（デフォーカスブラー）を適用したレイ原点
        float3 ray_origin = CameraCenter;
        if (DefocusAngle > 0.0f)
        {
            float3 disk_p = random_in_unit_disk(rng);
            ray_origin = CameraCenter + (disk_p.x * DefocusDiskU) + (disk_p.y * DefocusDiskV);
        }

        Ray cur_ray;
        cur_ray.origin = ray_origin;
        cur_ray.direction = normalize(pixel_sample - ray_origin);
        cur_ray.time = rand_float(rng); // シャッター時間[0,1]をサンプリング

        float3 throughput = float3(1.0f, 1.0f, 1.0f);
        float3 sample_color = float3(0.0f, 0.0f, 0.0f);

        // 反復パストレース（再帰の代わりにループを使用
        for (int bounce = 0; bounce < MaxDepth; ++bounce)
        {
            HitRecord rec = (HitRecord)0;   // ゼロ初期化
            if (hit_world(cur_ray, 0.001f, 1e9f, rng, rec))
            {
                // 1回目のバウンス（カメラから直接見える面）の座標を記録
                if (bounce == 0 && s == 0)
                {
                    first_hit_pos = rec.p;
                    first_hit_normal = rec.normal;
                    // マテリアルのアルベド輝度を計算（チェッカー境界保護用）
                    float3 alb = get_material_albedo(rec.mat, rec.p, rec.uv);
                    // 人間が感じる「明るさ（輝度：Luminance）」を算出するための標準的な重み係数（ITU-R BT.601 規格）
                    first_hit_albedo_lum = dot(alb, float3(0.299f, 0.587f, 0.114f));
                    is_first_hit_specular = (rec.mat.type == 1 || rec.mat.type == 2);
                }
                
                // マテリアルの発光成分を加算
                sample_color += throughput * emitted(rec);
                
                ScatterRecord srec;
                if (!scatter(cur_ray, rec, rng, srec))
                {
                    // DiffuseLightなどは散乱しないので終了
                    break;
                }
                
                if (srec.is_specular)
                {
                    throughput *= srec.attenuation;
                    cur_ray = srec.scattered_ray;
                }
                else
                {
                    // ----------------------------------------------------
                    // モンテカルロ積分: throughput *= (BRDF * cos(θ)) / PDF
                    // ----------------------------------------------------
                    float3 outward_dir = rec.front_face ? rec.normal : -rec.normal;
                    ONB onb = build_onb(outward_dir);
                    
                    float3 scattered_dir;
                    
                    // 1.散乱方向の生成（50%の確率で光源をサンプリング）
                    if (NumLightTargets >= 0 && rand_float(rng) < 0.5f)
                    {
                        scattered_dir = sample_light_targets(rec.p, rng);
                    }
                    else
                    {
                        float3 local_dir = random_cosine_direction(rng);
                        scattered_dir = normalize(onb_local(onb, local_dir));
                    }
                    
                    // 2.混合PDF値の計算（両方の確率密度を重み付け合算）
                    float p_cosine = max(0.0f, dot(outward_dir, scattered_dir) / PI);
                    float pdf_val = p_cosine;

                    if (NumLightTargets > 0)
                    {
                        float p_light = light_targets_pdf_value(rec.p, scattered_dir);
                        pdf_val = 0.5f * p_cosine + 0.5f * p_light;
                    }
                    
                    // 3.スループット更新
                    float cos_theta = dot(outward_dir, scattered_dir);
                    if (pdf_val > 1e-8f && cos_theta > 0.0f) // ゼロ除算禁止
                    {
                        // LambertianのBRDF(albedo / π)
                        float3 brdf = srec.attenuation / PI;
                        throughput *= (brdf * cos_theta) / pdf_val;

                        cur_ray.origin = rec.p + outward_dir * normal_buffer;
                        cur_ray.direction = scattered_dir;
                    }
                    else
                    {
                        break;
                    }
                }
                
                // ----------------------------------------------------
                // ロシアンルーレット (Russian Roulette) による動的枝刈り
                // ----------------------------------------------------
                // 最初の3バウンスは主照明や直接反射を確実に捉えるため無条件に通過させる
                if (bounce >= 3)
                {
                    // 現在の光の最大チャンネル成分を生存確率とする
                    float p_rr = max(throughput.r, max(throughput.g, throughput.b));

                    // 生存確率が極端に低くなった場合の重み爆発（ホタルノイズ）を防ぐクランプ
                    p_rr = clamp(p_rr, 0.005f, 0.95f);
                    
                    // サイコロを降って脱落判定
                    if (rand_float(rng) > p_rr)
                    {
                        // 追跡を早期終了
                        break;
                    }

                    // 生き残ったレイの重みを補正（期待値を維持）
                    throughput /= p_rr;
                }
            }
            else
            {
                // 空に到達
                if (BackgroundColor.x < 0.0f)
                {
                    float a = 0.5f * (cur_ray.direction.y + 1.0f);
                    float3 sky_color = lerp(float3(1.0f, 1.0f, 1.0f), float3(0.5f, 0.7f, 1.0f), a);
                    sample_color += throughput * sky_color; 
                }
                else
                {
                    // 指定された背景色
                    sample_color += throughput * BackgroundColor;
                }
                break;
            }
        }
        pass_color += sample_color;
    }

    // 初回パスなら0、2回目以降は前回の蓄積値を読みだして加算
    float3 prev_color = (PassIndex == 0) ? float3(0.0f, 0.0f, 0.0f) : OutputTexture[id.xy].rgb;
    // 線形色（リニアカラー）のまま出力バッファへ書き込む
    OutputTexture[id.xy] = float4(prev_color + pass_color, 1.0f);
    
    // ワールド座標の書き込み
    WorldPosTexture[id.xy] = float4(first_hit_pos, (float) is_first_hit_specular);
    
    // 法線とアルベド輝度を出力
    NormalTexture[id.xy] = float4(first_hit_normal, first_hit_albedo_lum);
}
