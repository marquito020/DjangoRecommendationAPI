#!/usr/bin/env python
"""
Ejemplo de uso de la API de recomendación de productos
"""
import requests
import json
import argparse

def train_model():
    """Entrena el modelo de recomendación"""
    url = "http://localhost:8000/api/train/"
    response = requests.get(url)
    if response.status_code == 200:
        print("Modelo entrenado correctamente")
        print(f"Productos disponibles: {response.json()['products']}")
    else:
        print(f"Error: {response.json()}")

def get_recommendations(input_products):
    """Obtiene recomendaciones para los productos de entrada"""
    url = "http://localhost:8000/api/recommendations/"
    data = {"input": input_products}
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Input: {result['input']}")
        print(f"Productos recomendados: {result['suggested']}")
    else:
        print(f"Error: {response.json()}")

def main():
    parser = argparse.ArgumentParser(description='Cliente para API de recomendación de productos')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--recommend', nargs=2, type=int, help='Obtener recomendaciones para dos productos')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.recommend:
        get_recommendations(args.recommend)
    else:
        print("Uso:")
        print("  Para entrenar el modelo: python app.py --train")
        print("  Para obtener recomendaciones: python app.py --recommend 1001 1003")

if __name__ == "__main__":
    main() 