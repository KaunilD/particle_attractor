#include "gameobject/gameobject.hpp"

GameObject::GameObject() {
	m_modelMatrix = glm::mat4x4(1.0);
};

GameObject::GameObject(bool t_npc) {
	m_npc = t_npc;
	
	m_modelMatrix = glm::mat4x4(1.0);
};

GameObject::~GameObject() {
	LOG("GameObject::Destroyed")
}

void GameObject::setMass(float t_mass) {
	m_mass = t_mass;
}

void GameObject::setMesh(shared_ptr<Mesh> t_mesh) {
	m_mesh = t_mesh;
}
/*
void GameObject::setMaterial(shared_ptr<Material> t_material) {
	m_material = t_material;
}
*/
void GameObject::setColor(glm::vec3 t_color) {
	m_color = t_color;
}

void GameObject::setPosition(glm::vec3 t_position) {
	m_position = t_position;
}

void GameObject::setScale(glm::vec3 t_scale) {
	m_scale = t_scale;
}

void GameObject::setupModelMatrix(glm::vec3 t_scale, glm::vec3 t_position) {
	m_modelMatrix = glm::mat4x4(1.0f);

	setScale(t_scale);
	setPosition(t_position);
}


float GameObject::getForceVector(const GameObject& p) const {
	glm::vec3 distance = p.getPosition() - m_position;
	float magnitude = sqrt(pow(distance.x, 2) + pow(distance.y, 2) + pow(distance.z, 2)) + 0.15;
	magnitude = fmin(20.f, fmax(magnitude, 5.f));
	return ( g * m_mass * p.m_mass) / (magnitude*magnitude * magnitude);
}


void GameObject::applyForceVector(float force, glm::vec3 distance) {
	glm::vec3 resForce = glm::normalize(distance)*force;
	acceleration += (resForce / m_mass);
	velocity += acceleration * (1 / 60.f);
	m_position += velocity * (1 / 60.f);
	acceleration *= 0;

	updateModelMatrix();
}

void GameObject::updateModelMatrix() {
	m_modelMatrix = glm::mat4x4(1.0f);
	m_modelMatrix = glm::scale(m_modelMatrix, m_scale);
	m_modelMatrix = glm::translate(m_modelMatrix, m_position);
}

void GameObject::setVelocity(glm::vec3 const &_velocity) {
	velocity = _velocity;
}
void GameObject::setAcceleration(glm::vec3 const &_acceleration) {
	acceleration = _acceleration;
}
