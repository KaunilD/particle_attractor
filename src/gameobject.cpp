#include "gameobject/gameobject.hpp"

GameObject::GameObject() {}



GameObject::GameObject(
	bool t_npc, float t_mass, std::shared_ptr<Mesh> t_mesh, std::shared_ptr<Material> t_material
):	npc(t_npc), mass(t_mass)
{	
	initializeOpenGLFunctions();
	float scale = qBound(7.f, mass*10, 10.0f);

	m_mesh = t_mesh;
	m_material = t_material;

	setupModelMatrix(
		position,
		QVector3D(scale, scale, scale)
	);
}

GameObject::~GameObject() {
	qDebug()<<"GameObject::Destroyed";
}

void GameObject::setColor(QVector3D _color) {
	color = _color;
}

void GameObject::setPosition(QVector3D _position) {
	position = _position;
}

void GameObject::setTranslate(QVector3D translate) {
	modelMatrix->translate(translate);
}

void GameObject::setScale(QVector3D _scale) {
	scale = _scale;
	modelMatrix->scale(_scale);
}

void GameObject::setupModelMatrix(QVector3D _translate, QVector3D _scale) {
	modelMatrix = make_unique<QMatrix4x4>();
	modelMatrix->setToIdentity();

	setScale(_scale);
	setTranslate(_translate);
}

void GameObject::updateObject(int frames, QKeyEvent * event) {

}


float GameObject::getForceVector(GameObject * p) {
	QVector3D distance = p->getPosition() - position;
	float magnitude = sqrt(pow(distance.x(), 2) + pow(distance.y(), 2) + pow(distance.z(), 2));
	magnitude = qBound(5.f, magnitude, 20.f);
	return ( g * mass * p->mass) / (magnitude * magnitude);
}


void GameObject::applyForceVector(float force, QVector3D distance) {
	QVector3D resForce = (distance.normalized())*force;
	acceleration += (resForce / mass);
	velocity += acceleration*(1/60.f);
	position += velocity * (1 / 60.f);
	acceleration *= 0;
	setupModelMatrix(position, scale);
}

void GameObject::setVelocity(QVector3D const &_velocity) {
	velocity = _velocity;
}
void GameObject::setAcceleration(QVector3D const &_acceleration) {
	acceleration = _acceleration;
}

void GameObject::setMass(float _mass) {
	mass = _mass;
}