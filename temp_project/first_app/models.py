from django.db import models

class User(models.Model):
    id = models.BigAutoField(primary_key=True)
    first_name = models.CharField(max_length=264, unique=False)
    last_name = models.CharField(max_length=264, unique=False)
    email = models.CharField(max_length=264, unique=False)
    birth_date = models.DateField()

    def __str__(self):
        return self.first_name