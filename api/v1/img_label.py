from .helpers import *

class ImgLabelApi():

    def create(request):
        try:
            params = get_params(request)
            if not params.get('name'): return custom_error(1, 'name is not null')
            obj = ImgLabel(name=params['name'], remake=params['remake'])
            if params.get('label_id'): obj.label_id = params.get('label_id')
            obj.save()
            return custom_success(obj.to_json())
        except Exception as e:
            return custom_error(1, str(e))

    def search(request):
        params = get_params(request)
        objs = ImgLabel.objects.all()
        total = objs.count()
        if params.get('label_id'): objs = ImgLabel.objects.filter(label_id=params['label_id'])
        if params.get('name'): objs = ImgLabel.objects.filter(name=params['name'])
        return custom_success({'total': total, 'list': pg(objs, params).to_json()})

    def get_all_options(request):
        objs = ImgLabel.objects.all()
        options = []
        for obj in objs:
            options.append({'label': obj.name, 'value': obj.id})
        return custom_success(options)

    def remove(request):
        params = get_params(request)
        model = ImgLabel.objects.filter(id=params['id'])
        model.delete()
        return custom_success()

    def update(request):
        params = get_params(request)
        obj = ImgLabel.objects.get(id=params['id'])
        if not obj: return custom_error(1, '没有找到这个id %s' % params['id'])
        obj.name = params['name']
        obj.remake = params['remake']
        obj.label_id = params['label_id']
        obj.save()
        return custom_success()
