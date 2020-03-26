#include "gameobject/gameobject.hpp"

GameObject::GameObject() {
	m_modelMatrix = make_unique<QMatrix4x4>();
	m_modelMatrix->setToIdentity();
};

GameObject::GameObject(bool t_npc) {
	m_npc = t_npc;
	
	m_modelMatrix = make_unique<QMatrix4x4>();
	m_modelMatrix->setToIdentity();
};

GameObject::~GameObject() {
	qDebug()<<"GameObject::Destroyed";
}

void GameObject::setMass(float t_mass) {
	m_mass = t_mass;
}

void GameObject::setMesh(shared_ptr<Mesh> t_mesh) {
	m_mesh = t_mesh;
}

void GameObject::setMaterial(shared_ptr<Material> t_material) {
	m_material = t_material;
}

void GameObject::setColor(QVector3D t_color) {
	m_color = t_color;
}

void GameObject::setPosition(QVector3D t_position) {
	m_position = t_position;
}

void GameObject::setScale(QVector3D t_scale) {
	m_scale = t_scale;
}

void GameObject::setupModelMatrix(QVector3D t_scale, QVector3D t_position) {
	m_modelMatrix = make_unique<QMatrix4x4>();
	m_modelMatrix->setToIdentity();

	setScale(t_scale);
	setPosition(t_position);
}


float GameObject::getForceVector(const GameObject& p) const {
	QVector3D distance = p.getPosition() - m_position;
	float magnitude = sqrt(pow(distance.x(), 2) + pow(distance.y(), 2) + pow(distance.z(), 2)) + 0.15;
	magnitude = qBound(5.f, magnitude, 20.f);
	return ( g * m_mass * p.m_mass) / (magnitude*magnitude * magnitude);
}


void GameObject::applyForceVector(float force, QVector3D distance) {
	QVector3D resForce = (distance.normalized())*force;
	acceleration += (resForce / m_mass);
	velocity += acceleration * (1 / 60.f);
	m_position += velocity * (1 / 60.f);
	acceleration *= 0;

	updateModelMatrix();
}

void GameObject::updateModelMatrix() {
	m_modelMatrix->setToIdentity();
	m_modelMatrix->scale(m_scale);
	m_modelMatrix->translate(m_position);
}

void GameObject::setVelocity(QVector3D const &_velocity) {
	velocity = _velocity;
}
void GameObject::setAcceleration(QVector3D const &_acceleration) {
	acceleration = _acceleration;
}
