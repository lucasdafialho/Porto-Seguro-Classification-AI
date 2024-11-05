from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('modelo_classificacao.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    marca = request.args.get('marca')
    modelo = request.args.get('modelo')
    problema = request.args.get('problema')
    
    if not all([marca, modelo, problema]):
        return jsonify({'error': 'Todos os parâmetros (marca, modelo, problema) são obrigatórios.'}), 400
    
    features = [marca, modelo, problema]
    df_features = pd.DataFrame([features], columns=["Marca", "Modelo", "Problema"])
    
    try:
        prediction = model.predict(df_features)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
