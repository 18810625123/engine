from . import *

class ImgLabel(models.Model):
    label = models.ForeignKey('self', on_delete=models.CASCADE, null=True)
    name = models.CharField(max_length=100, unique=True, blank=False, null=False)
    remake = models.CharField(max_length=1000, null=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'img_lable'

    def to_json(self):
        return {
            'id': self.id,
            'label_name': (self.label.name if self.label else ''),
            'label_id': self.label_id,
            'name': self.name,
            'remake': self.remake,
            'created_at': self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            'updated_at': self.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
