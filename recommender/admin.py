from django.contrib import admin
from .models import ProductRecommendation

@admin.register(ProductRecommendation)
class ProductRecommendationAdmin(admin.ModelAdmin):
    """Admin for ProductRecommendation model"""
    list_display = ('id', 'input_display', 'recommendation_display', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('input_products', 'recommended_products')
    readonly_fields = ('created_at',)
    
    def input_display(self, obj):
        """Format input products for display"""
        return str(obj.input_products)
    input_display.short_description = 'Input Products'
    
    def recommendation_display(self, obj):
        """Format recommended products for display"""
        return str(obj.recommended_products)
    recommendation_display.short_description = 'Recommendations'
