#include "gameobject/gameobject.hpp"

class SunObject : public GameObject {
public:
	SunObject() : GameObject() {};
	SunObject(bool npc) : GameObject(npc){};
	
	virtual void updateObject(float dt, const shared_ptr<GameObject>& gameObject) override {
		updateModelMatrix();
		return;
	};

};