#include "gameobject/gameobject.hpp"

class SunObject : public GameObject {
public:
	SunObject() : GameObject() {};
	SunObject(bool npc) : GameObject(npc){};
	
	virtual void updateObject(int frames, QKeyEvent* event, const GameObject& gameObject) override {
		updateModelMatrix();
		return;
	};

};