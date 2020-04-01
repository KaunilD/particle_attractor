#version 440 core

uniform sampler2D textureSampler;

in vec4 vs_position;
in vec3 vs_color;
in vec3 vs_normal;
in vec2 vs_texture;
in vec3 vs_eye;

out vec4 gl_FragColor;


void main()
{

	float ambientStrength = 0.5;

	vec3 lightColor = vec3(1.0, 0.0, 1.0);
	
	vec3 ambient = ambientStrength * lightColor;

	// diffuse 
	vec3 norm = normalize(vs_normal);
	vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0) - vs_position.xyz);
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;

	// specular lighting        
	float specularStrength = 0.5;
	
	vec3 viewDir = normalize(vs_eye - vec3(0, 0, 10.f));
	vec3 reflectDir = reflect(-lightDir, norm);

	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
	vec3 specular = specularStrength * spec * lightColor;

	gl_FragColor = vec4(vs_color * (ambient + diffuse + specular), 1.0);
}


