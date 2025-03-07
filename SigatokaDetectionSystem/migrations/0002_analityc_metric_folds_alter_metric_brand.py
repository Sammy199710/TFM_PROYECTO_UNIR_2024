# Generated by Django 5.1.6 on 2025-02-26 05:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SigatokaDetectionSystem', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Analityc',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.TextField(blank=True, null=True)),
                ('average', models.FloatField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Analityc',
                'verbose_name_plural': 'Analiticas',
                'ordering': ['-created'],
            },
        ),
        migrations.AddField(
            model_name='metric',
            name='folds',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='metric',
            name='brand',
            field=models.CharField(blank=True, max_length=3, null=True),
        ),
    ]
