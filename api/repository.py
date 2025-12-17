class Repository:
    def __init__(self, session):
        self.db = session

    def get_all(self, model):
        return self.db.query(model).all()

    def get_by_id(self, model, item_id):
        return self.db.query(model).get(item_id)

    def create(self, item):
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item

    def delete(self, model, item_id):
        item = self.db.query(model).get(item_id)
        if item:
            self.db.delete(item)
            self.db.commit()
        return item
