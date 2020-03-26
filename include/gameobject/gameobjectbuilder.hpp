#include "gameobject/gameobject.hpp"

class GameObjectBuilder {
protected:
	unique_ptr<GameObject> m_gameObject;
public:
	~GameObjectBuilder() = default;
	GameObject* getGameObject() {
		return m_gameObject.release();
	}

	void create(bool npc) {
		m_gameObject = make_unique<GameObject>(npc);
	}

	virtual void setMesh() = 0;
	virtual void setMaterial() = 0;
	virtual void setPosition() = 0;
	virtual void setScale() = 0;


};