#version 440 core

uniform sampler2D textureSampler;

uniform highp vec3 cameraEye;

uniform highp vec3 lightPosition;
uniform highp vec3 lightColor;
uniform highp vec3 lightAmbient;


in vec4 vs_position;
in vec3 vs_normal;
in vec2 vs_texture;


out vec4 gl_FragColor;


void main()
{

	vec3 eye_vec = normalize(cameraEye - vs_position.xyz);
	
	vec3 light_dist = lightPosition - vs_position.xyz;
	vec3 light_vec = normalize(light_dist);
	vec3 normal = normalize(vs_normal);

	float normal_dot_light = max(dot(light_vec, normal), 0.0);
	float light_mod = 50.0 / (1.0 + 0.25 * dot(light_dist, light_dist));

	vec3 specular = vec3(0);
	if (normal_dot_light > 0.0) {
		vec3 half_vec = normalize(light_vec + eye_vec);
		specular = light_mod * lightColor.rgb;
	}

	vec3 ambient = lightAmbient;
	vec3 diffuse = light_mod * normal_dot_light * lightColor;

	gl_FragColor = sqrt(vec4(ambient + diffuse + specular, 1.0));
}


