// zig fmt: off
const global =
\\  const gamma: f32 = 2.2;
\\  const pi: f32 = 3.1415926;
\\
\\  fn saturate(x: f32) -> f32 {
\\      return clamp(x, 0.0, 1.0);
\\  }
\\
\\  fn fresnelSchlickRoughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
\\      return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(1.0 - cos_theta, 5.0);
\\  }
\\
\\  fn radicalInverseVdc(in_bits: u32) -> f32 {
\\      var bits = (in_bits << 16u) | (in_bits >> 16u);
\\      bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
\\      bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
\\      bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
\\      bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
\\      return f32(bits) * bitcast<f32>(0x2f800000);
\\  }
\\
\\  fn hammersley(idx: u32, n: u32) -> vec2<f32> {
\\      return vec2(f32(idx) / f32(n), radicalInverseVdc(idx));
\\  }
\\
\\  fn importanceSampleGgx(xi: vec2<f32>, roughness: f32, n: vec3<f32>) -> vec3<f32> {
\\      let alpha = roughness * roughness;
\\      let phi = 2.0 * pi * xi.x;
\\      let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
\\      let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
\\
\\      var h: vec3<f32>;
\\      h.x = sin_theta * cos(phi);
\\      h.y = sin_theta * sin(phi);
\\      h.z = cos_theta;
\\
\\      // This is Right-Handed coordinate system and works for upper-left UV coordinate systems.
\\      var up_vector: vec3<f32>;
\\      if (abs(n.y) < 0.999) {
\\          up_vector = vec3(0.0, 1.0, 0.0);
\\      } else {
\\          up_vector = vec3(0.0, 0.0, 1.0);
\\      }
\\      let tangent_x = normalize(cross(up_vector, n));
\\      let tangent_y = normalize(cross(n, tangent_x));
\\
\\      // Tangent to world space.
\\      return normalize(tangent_x * h.x + tangent_y * h.y + n * h.z);
\\  }
\\
\\  fn geometrySchlickGgx(cos_theta: f32, roughness: f32) -> f32 {
\\      let k = (roughness * roughness) * 0.5;
\\      return cos_theta / (cos_theta * (1.0 - k) + k);
\\  }
\\
\\  fn geometrySmith(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
\\      return geometrySchlickGgx(n_dot_v, roughness) * geometrySchlickGgx(n_dot_l, roughness);
\\  }
\\
;
pub const precompute_env_tex_vs =
\\  struct Uniforms {
\\      object_to_clip: mat4x4<f32>,
\\  }
\\  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) position: vec3<f32>,
\\  }
\\
\\  @vertex fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      output.position_clip = vec4(position, 1.0) * uniforms.object_to_clip;
\\      output.position = position;
\\      return output;
\\  }
;
pub const precompute_env_tex_fs =
\\  @group(0) @binding(1) var equirect_tex: texture_2d<f32>;
\\  @group(0) @binding(2) var equirect_sam: sampler;
\\
\\  fn sampleSphericalMap(v: vec3<f32>) -> vec2<f32> {
\\      var uv = vec2(atan2(v.z, v.x), asin(v.y));
\\      uv = uv * vec2(0.1591, 0.3183);
\\      uv = uv + vec2(0.5);
\\      return uv;
\\  }
\\
\\  @fragment fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> @location(0) vec4<f32> {
\\      let uv = sampleSphericalMap(normalize(position));
\\      let color = textureSampleLevel(equirect_tex, equirect_sam, uv, 0.0).xyz;
\\      return vec4(color, 1.0);
\\  }
;
pub const precompute_irradiance_tex_vs =
\\  struct Uniforms {
\\      object_to_clip: mat4x4<f32>,
\\  }
\\  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
\\
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) position: vec3<f32>,
\\  }
\\
\\  @vertex fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      output.position_clip = vec4(position, 1.0) * uniforms.object_to_clip;
\\      output.position = position;
\\      return output;
\\  }
;
pub const precompute_irradiance_tex_fs = global ++
\\  @group(0) @binding(1) var env_tex: texture_cube<f32>;
\\  @group(0) @binding(2) var env_sam: sampler;
\\
\\  @fragment fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> @location(0) vec4<f32> {
\\      let n = normalize(position);
\\
\\      // This is Right-Handed coordinate system and works for upper-left UV coordinate systems.
\\      var up_vector: vec3<f32>;
\\      if (abs(n.y) < 0.999) {
\\          up_vector = vec3(0.0, 1.0, 0.0);
\\      } else {
\\          up_vector = vec3(0.0, 0.0, 1.0);
\\      }
\\      let tangent_x = normalize(cross(up_vector, n));
\\      let tangent_y = normalize(cross(n, tangent_x));
\\
\\      var num_samples: i32 = 0;
\\      var irradiance = vec3(0.0);
\\
\\      for (var phi = 0.0; phi < 2.0 * pi; phi = phi + 0.025) {
\\          let sin_phi = sin(phi);
\\          let cos_phi = cos(phi);
\\
\\          for (var theta = 0.0; theta < 0.5 * pi; theta = theta + 0.025) {
\\              let sin_theta = sin(theta);
\\              let cos_theta = cos(theta);
\\
\\              // Point on a hemisphere.
\\              let h = vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
\\
\\              // Transform from tangent space to world space.
\\              let sample_vector = tangent_x * h.x + tangent_y * h.y + n * h.z;
\\
\\              let irr = textureSample(env_tex, env_sam, sample_vector).xyz * cos_theta * sin_theta;
\\
\\              irradiance = irradiance + irr;
\\              num_samples = num_samples + 1;
\\          }
\\      }
\\
\\      irradiance = pi * irradiance * vec3(1.0 / f32(num_samples));
\\      return vec4(irradiance, 1.0);
\\  }
;
const precompute_filtered_env_tex_common =
\\  struct Uniforms {
\\      object_to_clip: mat4x4<f32>,
\\      roughness: f32,
\\  }
\\  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
\\
;
pub const precompute_filtered_env_tex_vs = precompute_filtered_env_tex_common ++
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) position: vec3<f32>,
\\  }
\\
\\  @vertex fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      output.position_clip = vec4(position, 1.0) * uniforms.object_to_clip;
\\      output.position = position;
\\      return output;
\\  }
;
pub const precompute_filtered_env_tex_fs = global ++ precompute_filtered_env_tex_common ++
\\  @group(0) @binding(1) var env_tex: texture_cube<f32>;
\\  @group(0) @binding(2) var env_sam: sampler;
\\
\\  @fragment fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> @location(0) vec4<f32> {
\\      let roughness = uniforms.roughness;
\\      let n = normalize(position);
\\      let r = n;
\\      let v = r;
\\
\\      var prefiltered_color = vec3(0.0);
\\      var total_weight = 0.0;
\\      let num_samples = 4096u;
\\
\\      for (var sample_idx = 0u; sample_idx < num_samples; sample_idx += 1u) {
\\          let xi = hammersley(sample_idx, num_samples);
\\          let h = importanceSampleGgx(xi, roughness, n);
\\          let lvec = 2.0 * dot(v, h) * h - v;
\\          let l = normalize(2.0 * dot(v, h) * h - v);
\\          let n_dot_l = saturate(dot(n, l));
\\          var color = textureSample(env_tex, env_sam, lvec).xyz * n_dot_l;
\\          prefiltered_color += color;
\\          total_weight += n_dot_l;
\\      }
\\      return vec4(prefiltered_color / max(total_weight, 0.001), 1.0);
\\  }
;
pub const precompute_brdf_integration_tex_cs = global ++
\\  @group(0) @binding(0) var brdf_tex: texture_storage_2d<rgba16float, write>;
\\
\\  fn integrate(roughness: f32, n_dot_v: f32) -> vec2<f32> {
\\      var v: vec3<f32>;
\\      v.x = 0.0;
\\      v.y = n_dot_v; // cos
\\      v.z = sqrt(1.0 - n_dot_v * n_dot_v); // sin
\\
\\      let n = vec3(0.0, 1.0, 0.0);
\\
\\      var a = 0.0;
\\      var b = 0.0;
\\      let num_samples = 1024u;
\\
\\      for (var sample_idx = 0u; sample_idx < num_samples; sample_idx = sample_idx + 1u) {
\\          let xi = hammersley(sample_idx, num_samples);
\\          let h = importanceSampleGgx(xi, roughness, n);
\\          let l = normalize(2.0 * dot(v, h) * h - v);
\\
\\          let n_dot_l = saturate(l.y);
\\          let n_dot_h = saturate(h.y);
\\          let v_dot_h = saturate(dot(v, h));
\\
\\          if (n_dot_l > 0.0) {
\\              let g = geometrySmith(n_dot_l, n_dot_v, roughness);
\\              let g_vis = g * v_dot_h / (n_dot_h * n_dot_v);
\\              let fc = pow(1.0 - v_dot_h, 5.0);
\\              a = a + (1.0 - fc) * g_vis;
\\              b = b + fc * g_vis;
\\          }
\\      }
\\      return vec2(a, b) / vec2(f32(num_samples));
\\  }
\\
\\  @compute @workgroup_size(8, 8, 1)
\\  fn main(
\\      @builtin(global_invocation_id) global_id: vec3<u32>,
\\  ) {
\\      let dim = textureDimensions(brdf_tex);
\\      let roughness = f32(global_id.y + 1u) / f32(dim.y);
\\      let n_dot_v = f32(global_id.x + 1u) / f32(dim.x);
\\      let result = integrate(roughness, n_dot_v);
\\      textureStore(brdf_tex, vec2<i32>(global_id.xy), vec4(result, 0.0, 1.0));
\\  }
;
const mesh_common =
\\  struct MeshUniforms {
\\      object_to_world: mat4x4<f32>,
\\      world_to_clip: mat4x4<f32>,
\\      camera_position: vec3<f32>,
\\      draw_mode: i32,
\\      emissive_color: vec3<f32>,
\\      _pad: f32,
\\  }
\\  @group(0) @binding(0) var<uniform> uniforms: MeshUniforms;
\\
;
pub const mesh_vs = mesh_common ++
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) position: vec3<f32>,
\\      @location(1) normal: vec3<f32>,
\\      @location(2) texcoord: vec2<f32>,
\\      @location(3) tangent: vec4<f32>,
\\  }
\\
\\  @vertex fn main(
\\      @location(0) position: vec3<f32>,
\\      @location(1) normal: vec3<f32>,
\\      @location(2) texcoord: vec2<f32>,
\\      @location(3) tangent: vec4<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      output.position_clip = vec4(position, 1.0) * uniforms.object_to_world * uniforms.world_to_clip;
\\      output.position = (vec4(position, 1.0) * uniforms.object_to_world).xyz;
\\      output.normal = normal;
\\      output.texcoord = texcoord;
\\      output.tangent = tangent;
\\      return output;
\\  }
;
pub const mesh_fs = global ++ mesh_common ++
\\  @group(0) @binding(1) var ao_tex: texture_2d<f32>;
\\  @group(0) @binding(2) var base_color_tex: texture_2d<f32>;
\\  @group(0) @binding(3) var metallic_roughness_tex: texture_2d<f32>;
\\  @group(0) @binding(4) var normal_tex: texture_2d<f32>;
\\
\\  @group(0) @binding(8) var aniso_sam: sampler;
\\
\\  struct GBufferOutput {
\\      @location(0) albedo_roughness: vec4<f32>,
\\      @location(1) normal_metal: vec4<f32>,
\\      @location(2) emissive: vec4<f32>,
\\  }
\\
\\  @fragment fn main(
\\      @location(0) position: vec3<f32>,
\\      @location(1) normal: vec3<f32>,
\\      @location(2) texcoord: vec2<f32>,
\\      @location(3) tangent: vec4<f32>,
\\  ) -> GBufferOutput {
\\      let unit_normal = normalize(normal);
\\      let unit_tangent = vec4(normalize(tangent.xyz), tangent.w);
\\      let unit_bitangent = normalize(cross(normal, unit_tangent.xyz)) * unit_tangent.w;
\\
\\      let object_to_world = mat3x3(
\\          uniforms.object_to_world[0].xyz,
\\          uniforms.object_to_world[1].xyz,
\\          uniforms.object_to_world[2].xyz,
\\      );
\\      var n = normalize(textureSample(normal_tex, aniso_sam, texcoord).xyz * 2.0 - 1.0);
\\      n = n * transpose(mat3x3(unit_tangent.xyz, unit_bitangent, unit_normal));
\\      n = normalize(n * object_to_world);
\\
\\      var metallic: f32;
\\      var roughness: f32;
\\      {
\\          let rm = textureSample(metallic_roughness_tex, aniso_sam, texcoord).yz;
\\          roughness = rm.x;
\\          metallic = rm.y;
\\      }
\\      let base_color = pow(textureSample(base_color_tex, aniso_sam, texcoord).xyz, vec3(gamma));
\\      
\\      var output: GBufferOutput;
\\      output.albedo_roughness = vec4(base_color, roughness);
\\      output.normal_metal = vec4(n, metallic);
\\      output.emissive = vec4(uniforms.emissive_color, 1.0);
\\      return output;
\\  }
;
pub const deferred_vs =
\\  @vertex fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
\\      var pos = array<vec2<f32>, 3>(
\\          vec2<f32>(-1.0, -1.0),
\\          vec2<f32>( 3.0, -1.0),
\\          vec2<f32>(-1.0,  3.0)
\\      );
\\      return vec4<f32>(pos[vertex_index], 0.0, 1.0);
\\  }
;
pub const deferred_fs = global ++
\\  struct DeferredUniforms {
\\      inverse_view_proj: mat4x4<f32>,
\\      camera_position: vec3<f32>,
\\      draw_mode: i32,
\\  }
\\  @group(0) @binding(0) var<uniform> uniforms: DeferredUniforms;
\\  @group(0) @binding(1) var g_albedo: texture_2d<f32>;
\\  @group(0) @binding(2) var g_normal: texture_2d<f32>;
\\  @group(0) @binding(3) var g_emissive: texture_2d<f32>;
\\  @group(0) @binding(4) var g_depth: texture_depth_2d;
\\
\\  @group(0) @binding(5) var irradiance_tex: texture_cube<f32>;
\\  @group(0) @binding(6) var filtered_env_tex: texture_cube<f32>;
\\  @group(0) @binding(7) var brdf_integration_tex: texture_2d<f32>;
\\  @group(0) @binding(8) var aniso_sam: sampler;
\\  @group(0) @binding(9) var g_sdf: texture_2d<f32>;
\\  @group(1) @binding(0) var g_gi: texture_2d<f32>;
\\
\\  @fragment fn main(
\\      @builtin(position) position: vec4<f32>,
\\  ) -> @location(0) vec4<f32> {
\\      let uv = vec2<i32>(position.xy);
\\      let albedo_rough = textureLoad(g_albedo, uv, 0);
\\      let normal_metal = textureLoad(g_normal, uv, 0);
\\      let emissive = textureLoad(g_emissive, uv, 0);
\\      let depth = textureLoad(g_depth, uv, 0);
\\      let gi = textureLoad(g_gi, uv, 0).xyz;
\\
\\      if (uniforms.draw_mode == 1) { // Albedo / Roughness
\\          return vec4(pow(albedo_rough.xyz, vec3(1.0 / gamma)), 1.0);
\\      } else if (uniforms.draw_mode == 2) { // Normal / Metallic
\\          return vec4(normal_metal.xyz * 0.5 + 0.5, 1.0);
\\      } else if (uniforms.draw_mode == 3) { // Emissive
\\          return vec4(emissive.xyz, 1.0);
\\      } else if (uniforms.draw_mode == 4) { // Depth
\\          return vec4(vec3(pow(depth, 100.0)), 1.0); // Visualize depth curve
\\      } else if (uniforms.draw_mode == 5) { // SDF
\\          // The SDF can be coarser than the framebuffer when the user
\\          // picks a non-Ultra SDF quality — sample with scaled coords.
\\          let sdf_dim = textureDimensions(g_sdf);
\\          let fb_dim = textureDimensions(g_depth);
\\          let sdf_uv = vec2<i32>(vec2<f32>(uv) * vec2<f32>(f32(sdf_dim.x), f32(sdf_dim.y)) / vec2<f32>(f32(fb_dim.x), f32(fb_dim.y)));
\\          let sdf_val = textureLoad(g_sdf, sdf_uv, 0).x;
\\          return vec4(vec3(sdf_val), 1.0);
\\      } else if (uniforms.draw_mode == 6) { // Radiance Cascades GI
\\          return vec4(gi, 1.0);
\\      }
\\
\\      if (depth >= 1.0) {
\\          return vec4(0.0, 0.0, 0.0, 0.0);
\\      }
\\
\\      let base_color = albedo_rough.xyz;
\\      let roughness = albedo_rough.w;
\\      let n = normalize(normal_metal.xyz);
\\      let metallic = normal_metal.w;
\\
\\      let dim = textureDimensions(g_depth);
\\      let ndc_x = (f32(uv.x) / f32(dim.x)) * 2.0 - 1.0;
\\      let ndc_y = 1.0 - (f32(uv.y) / f32(dim.y)) * 2.0;
\\      let clip_pos = vec4(ndc_x, ndc_y, depth, 1.0);
\\      var world_pos = uniforms.inverse_view_proj * clip_pos;
\\      world_pos = world_pos / world_pos.w;
\\
\\      let v = normalize(uniforms.camera_position - world_pos.xyz);
\\      let n_dot_v = saturate(dot(n, v));
\\
\\      let f0 = mix(vec3(0.04), base_color, vec3(metallic));
\\      let r = reflect(-v, n);
\\      let f = fresnelSchlickRoughness(n_dot_v, f0, roughness);
\\      let kd = (1.0 - f) * (1.0 - metallic);
\\
\\      let irradiance = textureSampleLevel(irradiance_tex, aniso_sam, n, 0.0).xyz;
\\      let prefiltered_color = textureSampleLevel(
\\          filtered_env_tex,
\\          aniso_sam,
\\          r,
\\          roughness * 5.0, // roughness * (num_mip_levels - 1.0)
\\      ).xyz;
\\      let env_brdf = textureSampleLevel(
\\          brdf_integration_tex,
\\          aniso_sam,
\\          vec2(min(n_dot_v, 0.999), roughness),
\\          0.0
\\      ).xy;
\\
\\      let diffuse = irradiance * base_color;
\\      let specular = prefiltered_color * (f * env_brdf.x + env_brdf.y);
\\      let ambient = kd * diffuse + specular; // Note: removed ao for now, can be sampled if G-Buffer stores it
\\
\\      // Screen-Space Radiance Cascades supplies the indirect diffuse
\\      // contribution from nearby on-screen surfaces; modulate it by
\\      // the albedo and the diffuse Fresnel term so metals stay metallic.
\\      let indirect_gi = gi * base_color * kd;
\\
\\      var color = ambient + emissive.xyz + indirect_gi;
\\      color = color / (color + vec3(1.0));
\\      return vec4(pow(color, vec3(1.0 / gamma)), 1.0);
\\  }
;
pub const sample_env_tex_vs =
\\  struct Uniforms {
\\      object_to_clip: mat4x4<f32>,
\\  }
\\  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
\\
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) position: vec3<f32>,
\\  }
\\
\\  @vertex fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      output.position_clip = (vec4(position, 1.0) * uniforms.object_to_clip).xyww;
\\      output.position = position;
\\      return output;
\\  }
;
pub const sample_env_tex_fs = global ++
\\  @group(0) @binding(1) var env_tex: texture_cube<f32>;
\\  @group(0) @binding(2) var env_sam: sampler;
\\
\\  @fragment fn main(
\\      @location(0) position: vec3<f32>,
\\  ) -> @location(0) vec4<f32> {
\\      var color = textureSample(env_tex, env_sam, position).xyz;
\\      color = color / (color + vec3(1.0));
\\      return vec4(pow(color, vec3(1.0 / gamma)), 1.0);
\\  }
;
// zig fmt: on
