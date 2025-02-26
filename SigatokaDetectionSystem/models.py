from django.db import models



class Metric(models.Model):
    epoch = models.IntegerField()
    accuracy = models.FloatField()
    loss = models.FloatField()
    brand = models.CharField(max_length=3, null=True, blank=True)
    epoch_brand = models.IntegerField( null=True, blank=True)
    folds = models.IntegerField( null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    created = models.DateTimeField(auto_now_add=True)  # Este campo se define para registrar cuando se crea el objeto
    updated = models.DateTimeField(auto_now=True)  # Este campo se actualiza cada vez que el objeto es guardado

    class Meta:
        verbose_name = "Metric"
        verbose_name_plural = "Metricas"
        ordering = ['-created']

    def __str__(self):
        return f"Epoch={self.epoch}: Acc={self.accuracy}, Loss={self.loss},"+\
               f"Brand={self.brand}, Epoch_Brand={self.epoch_brand}, Timestamp={self.timestamp}"+\
               f"Folds={self.folds}"
    
class Analityc(models.Model):
    description = models.TextField(null=True, blank=True)  # Sin límite de caracteres
    average = models.FloatField()
    brand = models.CharField(max_length=3, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    created = models.DateTimeField(auto_now_add=True)  # Este campo se define para registrar cuando se crea el objeto
    updated = models.DateTimeField(auto_now=True)  # Este campo se actualiza cada vez que el objeto es guardado
    
    class Meta:
        verbose_name = "Analityc"
        verbose_name_plural = "Analiticas"
        ordering = ['-created']

    def __str__(self):
        return f"description={self.description}: average={self.average}, timestamp={self.timestamp}"


class Confusion_Matrix(models.Model):
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    support = models.IntegerField() #Cantidad de datos
    class_label = models.CharField(max_length=50, null=True, blank=True) #Identificador de clases
    brand = models.CharField(max_length=3, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    created = models.DateTimeField(auto_now_add=True)  # Este campo se define para registrar cuando se crea el objeto
    updated = models.DateTimeField(auto_now=True)  # Este campo se actualiza cada vez que el objeto es guardado
    
    class Meta:
        verbose_name = "Confusion_Matrix"
        verbose_name_plural = "Matriz de Confusion"
        ordering = ['-created']

    def __str__(self):
        return f"clases={self.class_label}, precision={self.precision}, recall={self.recall}, f1_score={self.f1_score}, support={self.support}, timestamp={self.timestamp}"