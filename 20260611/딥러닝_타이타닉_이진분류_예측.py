from tensorflow.keras.models import load_model # type: ignore

titanic_bestmodel = load_model("/home/sm/tf_env/20260611/titanic_bestmodel.keras")
titanic_bestmodel.summary()