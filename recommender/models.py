from django.db import models

# Create your models here.

class ProductRecommendation(models.Model):
    """
    Modelo para almacenar las recomendaciones de productos
    
    Guarda los productos de entrada y los productos recomendados
    """
    input_products = models.JSONField(
        help_text="Lista de IDs de productos de entrada (formato JSON)"
    )
    recommended_products = models.JSONField(
        help_text="Lista de IDs de productos recomendados (formato JSON)"
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Fecha y hora en que se generó la recomendación"
    )
    
    class Meta:
        verbose_name = "Recomendación de Productos"
        verbose_name_plural = "Recomendaciones de Productos"
        ordering = ['-created_at']
        
    def __str__(self):
        return f"Recomendación para {self.input_products}"
