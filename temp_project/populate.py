import  os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'temp_project.settings')

import django
django.setup()

import random
from first_app.models import User
from faker import Faker

fakegen= Faker()

def populate(N=5):

    for entry in range(N):
        fake_first_name = fakegen.first_name()
        fake_last_name = fakegen.last_name()
        fake_email = fakegen.email()
        fake_birth_date = fakegen.date_of_birth()

        users = User.objects.get_or_create(first_name = fake_first_name,
                                           last_name = fake_last_name,
                                           email = fake_email,
                                           birth_date = fake_birth_date)[0]


if __name__ == '__main__':
    print('Populating data')
    populate(66)
    print('Populating completed')