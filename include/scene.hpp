#ifndef SCENE_H
#define SCENE_H

#include "libs.hpp"
#include "gameobject/mesh.hpp"

#include "cuda/update_gameobject.hpp"
#include "cuda/update_models.hpp"
#include "cuda/utils.hpp"

#include "algorithms/algorithms.hpp"

class Scene {
public:
	
	Scene();

	Scene(int H, int W) : m_numparticles(H*W), m_height(H), m_width(W)  {
		h_positions = new float4[H * W];
		h_velocities = new float4[H * W];
		h_masses = new float[H * W];

		CHECK(cudaMalloc(&d_positions, H * W * sizeof(float4)));
		CHECK(cudaMalloc(&d_velocities, H * W * sizeof(float4)));
		CHECK(cudaMalloc(&d_masses, H * W * sizeof(float)));

		cudaMemset(d_positions, H * W * sizeof(float4), 0);
		cudaMemset(d_velocities, H * W * sizeof(float4), 0);
		cudaMemset(d_masses, H * W * sizeof(float), 0);

	};

	void init() {
		float dx = 10 / (float)m_width;
		float dy = 10 / (float)m_height;
		int count = 0;
		for (float i = 0; i < 10; i+=dx) {
			for (float j = 0; j < 10; j+=dy) {
				h_positions[count] = make_float4(i - 5, j - 5, 1.0, 1.0f);

				h_velocities[count] = make_float4(
					Algorithms::randomInt(10) / 10 - 0.5,
					Algorithms::randomInt(10) / 10 - 0.5,
					Algorithms::randomInt(10) / 10 - 0.5,
					1.0f);

				h_masses[count] = Algorithms::randomInt(10) / 10.0f + 0.01;

				glm::mat4 mat(1.0f);
				mat = glm::translate(mat, glm::vec3(h_positions[count].x, h_positions[count].y, h_positions[count].z));
				h_models.push_back(mat);

				count++;
	
			}
		}
		
		cudaMemcpy(d_positions, h_positions, m_numparticles * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_velocities, h_velocities, m_numparticles * sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(d_masses, h_masses, m_numparticles * sizeof(float), cudaMemcpyHostToDevice);
	}

	void update() {
		/*
			Upadte positions.
			gravity calculations performed here.
		*/
		launch_kernel_update(
			d_positions, d_velocities, 
			h_positions, h_velocities, 
			d_masses, 
			m_numparticles
		);

		/*
			Setup draw for each mesh
		*/
		for (int i = 0; i < m_meshes.size(); i++) {
			/*
				Get latest device ptr for OGL buffer.
			*/
			m_meshes.at(i)->getCUDAptr();
			/*
				Update model matrices
			*/
			launch_kernel_models(
				d_positions,
				m_meshes.at(i)->d_modelBuffer, 
				m_numparticles
			);
			/*
				Draw!
			*/
			m_meshes.at(i)->drawInstanced(m_numparticles);
		}

	};

	void addMesh(shared_ptr<Mesh> t_mesh) {
		t_mesh->prepareInstances(h_models);
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
	int m_height, m_width;
	float4* d_positions, * h_positions;
	float4* d_velocities, * h_velocities;
	float* d_masses, * h_masses;
	std::vector<glm::mat4> h_models;
	std::vector<shared_ptr<Mesh>> m_meshes;
};

#endif SCENE_H