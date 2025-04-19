from rest_framework import serializers
from .models import ProductRecommendation

class RecommendationInputSerializer(serializers.Serializer):
    """
    Serializer para la entrada de recomendaciones
    
    Espera una lista de exactamente 2 IDs de productos
    """
    input = serializers.ListField(
        child=serializers.IntegerField(),
        help_text="Lista de IDs de productos para generar recomendaciones (exactamente 2 productos)",
        min_length=2,
        max_length=2
    )

class RecommendationOutputSerializer(serializers.Serializer):
    """
    Serializer para la salida de recomendaciones
    
    Devuelve los productos de entrada y los productos recomendados
    """
    input = serializers.ListField(
        child=serializers.IntegerField(),
        help_text="Lista de IDs de productos de entrada"
    )
    suggested = serializers.ListField(
        child=serializers.IntegerField(),
        help_text="Lista de IDs de productos recomendados"
    )

class ProductRecommendationSerializer(serializers.ModelSerializer):
    """
    Serializer para el modelo ProductRecommendation
    
    Utilizado para almacenar un historial de recomendaciones
    """
    class Meta:
        model = ProductRecommendation
        fields = ['id', 'input_products', 'recommended_products', 'created_at']
        read_only_fields = ['created_at'] 