#ifndef GAMEOBJECT_H
#define GAMEOBJECT_H

#include "libs.hpp"
#include "vertex.hpp"
#include "shaderprogram.hpp"
#include "gameobject/mesh.hpp"
#include "gameobject/material.hpp"

class GameObject{

public:
	GameObject();
	GameObject(bool);
	~GameObject();

	bool m_npc;
	glm::vec3 velocity = glm::vec3(1.0f, 2.0f, 1.0f);
	glm::vec3 acceleration = glm::vec3(0.0f);
	float m_mass, g = 0.4f, angle = 0, angularVelocity = 0, angularAcceleration = 0.11;


	glm::mat4x4 m_modelMatrix;
	glm::vec3 m_position, m_color, m_scale;

	shared_ptr<Mesh> m_mesh;
	//shared_ptr<Material> m_material;

	glm::mat4x4 getModelMatrix() const { return m_modelMatrix; };

	void setupModelMatrix(glm::vec3 translate, glm::vec3 scale);
	void updateModelMatrix();

	void setMass(float t_mass);
	void setMesh(shared_ptr<Mesh> t_mesh);
	//void setMaterial(shared_ptr<Material> t_material);
	void setScale(glm::vec3 scale);
	void setPosition(glm::vec3 translate);


	void setColor(glm::vec3 color);
	glm::vec3 getColor() { return m_color; };

	glm::vec3 getPosition() const { return m_position; };

	void updatePosition(glm::vec3);
	void setVelocity(glm::vec3  const &);
	void setAcceleration(glm::vec3 const &);

	float getForceVector(const shared_ptr<GameObject>& p) const;
	void applyForceVector(float, glm::vec3, float);

	virtual void updateObject(float dt, const shared_ptr<GameObject>& referenceObj) = 0;

};
#endif