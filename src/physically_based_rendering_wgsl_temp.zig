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
;

const mesh_common =
\\  struct MeshUniforms {
\\      object_to_world: mat4x4<f32>,
\\      world_to_clip: mat4x4<f32>,
\\      camera_position: vec3<f32>,
\\      draw_mode: i32,
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
\\  @group(0) @binding(8) var aniso_sam: sampler; // Kept at binding 8 for compatibility with old layout for now
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
\\      output.emissive = vec4(0.0, 0.0, 0.0, 1.0); // Emissive could be sampled if available
\\      return output;
\\  }
;

pub const deferred_vs =
\\  @vertex fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
\\      var x = f32(i32(vertex_index) - 1);
\\      var y = f32(i32(vertex_index & 1u) * 2 - 1);
\\      return vec4(x, y, 0.0, 1.0);
\\  }
;

pub const deferred_fs = global ++
\\  struct DeferredUniforms {
\\      camera_position: vec3<f32>,
\\      pad: f32,
\\  }
\\  @group(0) @binding(0) var<uniform> uniforms: DeferredUniforms;
\\  @group(0) @binding(1) var g_albedo: texture_2d<f32>;
\\  @group(0) @binding(2) var g_normal: texture_2d<f32>;
\\  @group(0) @binding(3) var g_emissive: texture_2d<f32>;
\\  @group(0) @binding(4) var g_depth: texture_2d<f32>;
\\
\\  @group(0) @binding(5) var irradiance_tex: texture_cube<f32>;
\\  @group(0) @binding(6) var filtered_env_tex: texture_cube<f32>;
\\  @group(0) @binding(7) var brdf_integration_tex: texture_2d<f32>;
\\  @group(0) @binding(8) var aniso_sam: sampler;
\\
\\  // TODO: Reconstruct world position from depth
\\
\\  @fragment fn main(
\\      @builtin(position) position: vec4<f32>,
\\  ) -> @location(0) vec4<f32> {
\\      let uv = vec2<i32>(position.xy);
\\      let albedo_rough = textureLoad(g_albedo, uv, 0);
\\      let normal_metal = textureLoad(g_normal, uv, 0);
\\      let emissive = textureLoad(g_emissive, uv, 0);
\\      let depth = textureLoad(g_depth, uv, 0).x;
\\
\\      if (depth >= 1.0) {
\\          // Background
\\          return vec4(0.0, 0.0, 0.0, 1.0);
\\      }
\\
\\      let base_color = albedo_rough.xyz;
\\      let roughness = albedo_rough.w;
\\      let n = normalize(normal_metal.xyz);
\\      let metallic = normal_metal.w;
\\
\\      // Very simple ambient light for now since we haven't reconstructed position.
\\      // Will expand this to full PBR.
\\      let irradiance = textureSample(irradiance_tex, aniso_sam, n).xyz;
\\      let diffuse = irradiance * base_color;
\\
\\      var color = diffuse;
\\      color = color / (color + vec3(1.0));
\\      return vec4(pow(color, vec3(1.0 / gamma)), 1.0);
\\  }
;
// zig fmt: on
