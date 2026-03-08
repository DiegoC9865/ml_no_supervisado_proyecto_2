
import os
import joblib
from src.singleton import SingletonMeta

class ModelController(metaclass=SingletonMeta):
    value: str = None
    def __init__(self):
        print("init model contorller")
        self.model_path = os.path.join("models","model.joblib")
        self.model = joblib.load(self.model_path)

        self.ods_dict = {
            1: "Fin de la pobreza",
            2: "Hambre cero",
            3: "Salud y bienestar",
            4: "Educación de calidad",
            5: "Igualdad de género",
            6: "Agua limpia y saneamiento",
            7: "Energía asequible y no contaminante",
            8: "Trabajo decente y crecimiento económico",
            9: "Industria, innovación e infraestructura",
            10: "Reducción de las desigualdades",
            11: "Ciudades y comunidades sostenibles",
            12: "Producción y consumo responsables",
            13: "Acción por el clima",
            14: "Vida submarina",
            15: "Vida de ecosistemas terrestres",
            16: "Paz, justicia e instituciones sólidas",
            17: "Alianzas para lograr los objetivos"
        }

    def predict (self, data:list) -> list[int]:
        """
        ya que el modelo es como tal un pipeline de transformación, no es necesario
        implementar meodos de preprocesado,
        """
        print("model controller predicting")
        return self.model.predict(data)
    
    def name_mapper (self, ods:int) -> str:
        return self.ods_dict.get(ods, "Error") 



if __name__ == '__main__':
    text = 'Las algas marinas como laver, la mostaza marina y las algas marinas representan alrededor del 70 % de la producción acuícola total (Panel A). Los principales destinos de las exportaciones fueron Japón, la República Popular China (en adelante, "China"), Tailandia y los Estados Unidos. Los mercados tradicionales como Japón y China se debilitaron, pero esto fue más que compensado por aumentos en mercados más nuevos como Estados Unidos y la Unión Europea. Las Transferencias Financieras del Gobierno a servicios generales disminuyeron un 42% (Panel C).'

    controller = ModelController()
    predictions = controller.predict([text])
    print(controller.name_mapper(predictions[0]))