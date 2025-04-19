from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
import json
from .recommendation import recommendation_system

class RecommendationAPITests(TestCase):
    def setUp(self):
        self.client = APIClient()
        # Ensure model is trained
        recommendation_system.train()
    
    def test_get_recommendations(self):
        """Test the recommendation API endpoint"""
        url = reverse('get_recommendations')
        data = {"input": [1001, 1003]}
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('input', response.data)
        self.assertIn('suggested', response.data)
        self.assertEqual(response.data['input'], [1001, 1003])
        self.assertIsInstance(response.data['suggested'], list)
    
    def test_invalid_input(self):
        """Test the recommendation API endpoint with invalid input"""
        url = reverse('get_recommendations')
        data = {"input": [1001]}  # Only one product
        response = self.client.post(url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_train_model(self):
        """Test the training API endpoint"""
        url = reverse('train_model')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('message', response.data)
        self.assertIn('products', response.data)
