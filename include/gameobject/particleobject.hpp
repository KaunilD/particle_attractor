#include "gameobject/gameobject.hpp"

class ParticleObject : public GameObject {
public:
	ParticleObject() :GameObject() {};
	ParticleObject(bool t_npc) : GameObject(t_npc) {};


	virtual void updateObject(float dt, const shared_ptr<GameObject>& gameObject) override {
		float force = getForceVector(gameObject);
		this->applyForceVector(force, gameObject->getPosition() - this->getPosition(), dt);
	}

};