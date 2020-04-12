#ifndef SCENE_H
#define SCENE_H

#include "libs.hpp"
#include "gameobject/mesh.hpp"

#include "cuda/update_gameobject.hpp"
#include "cuda/update_models.hpp"
#include "algorithms/algorithms.hpp"

class Scene {
public:
	
	Scene();

	Scene(int t_numparticles) : m_numparticles(t_numparticles) {
		h_positions = new float4[t_numparticles];
		h_velocities = new float4[t_numparticles];
		h_masses = new float[t_numparticles];
		
		cudaMalloc(&d_positions, t_numparticles * sizeof(float4));
		cudaMalloc(&d_velocities, t_numparticles * sizeof(float4));
		cudaMalloc(&d_masses, t_numparticles * sizeof(float));

	};

	void init() {
		for (int i = 0; i < m_numparticles; i++) {
			h_positions[i] = make_float4(
				Algorithms::randomInt(10) / 100.0f,
				Algorithms::randomInt(10) / 100.0f,
				Algorithms::randomInt(10) / 100.0f,
				1.0f
			);

			h_velocities[i] = make_float4(
				Algorithms::randomInt(10) / 10.0f,
				Algorithms::randomInt(10) / 10.0f,
				Algorithms::randomInt(10) / 10.0f,
				1.0f
			);
			
			h_masses[i] = Algorithms::randomInt(10) / 10.0f + 0.001;

			glm::mat4x4 mat = glm::mat4x4(1.0f);
			mat = glm::translate(mat, glm::vec3(h_positions[i].x, h_positions[i].y, h_positions[i].z));
			mat = glm::scale(mat, glm::vec3(1.0f));
			h_models.push_back(mat);
		}

		cudaMemcpy(d_positions, h_positions, m_numparticles * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_velocities, h_velocities, m_numparticles * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_masses, h_masses, m_numparticles * sizeof(float), cudaMemcpyHostToDevice);

	}

	void update() {

		launch_kernel_update(
			d_positions, d_velocities, 
			h_positions, h_velocities, 
			d_masses, 
			m_numparticles);
		m_meshes.at(0)->getCUDAptr();

		launch_kernel_models(
			d_positions,
			m_meshes.at(0)->d_modelBuffer, 
			m_meshes.at(0)->num_bytes);

		m_meshes.at(0)->drawInstanced(m_numparticles);
	};

	void addMesh(shared_ptr<Mesh> t_mesh) {
		t_mesh->prepareInstances(h_models, m_numparticles);
		m_meshes.push_back(t_mesh);
	}

	~Scene() {
		cudaFree(d_positions);
		cudaFree(d_velocities);
		cudaFree(d_masses);

		delete[] h_positions;
		delete[] h_velocities;
		delete[] h_masses;
	};

	int m_numparticles;
	float4* d_positions, * h_positions;
	float4* d_velocities, * h_velocities;
	float* d_masses, * h_masses;
	std::vector<glm::mat4x4> h_models;

	std::vector<shared_ptr<Mesh>> m_meshes;
};

#endif SCENE_H