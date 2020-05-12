from .helpers import *

class ImgApi():

    def zoomed(request):
        params = get_params(request)
        img = Img()
        img.read_url(params['img_url'])
        img.zoomed(w=params['w'], h=params['h'])
        return custom_success({
            "base64": img.to_base64(img.zoomed_img)
        })

    def search(request):
        params = get_params(request)
        img = Img.objects.filter(md5=params['md5'])
        if not img: return custom_error(1, "没有找到 %s 这个图片" % (params['md5']))
        img.delete()
        return custom_success()

    def remove(request):
        params = get_params(request)
        img = Img.objects.filter(md5=params['md5'])
        if not img: return custom_error(1, "没有找到 %s 这个图片" % (params['md5']))
        img.delete()
        return custom_success()

    def reshape(request):
        params = get_params(request)
        img = Img(url=params['img_url'])
        img.read_url()
        img.resize((1000, 500))
        return custom_success({
            'old_img': {
                'shape': img.image.shape
            },
            'new_img': {
                'shape': img.resize_img.shape,
                'base64': img.to_base64(img.resize_img)
            }
        })

    def handler(request):
        params = get_params(request)
        img = Img(url=params['img_url'])
        img.read_url()
        img.gray()
        img.filter2d(kernel=params['kernel'])
        img.threshold(min=params['min'], max=params['max'])
        img.gauss(5, 5)
        img.canny(120, 240)
        return custom_success({
            "old_img": {
                "shape": img.image.shape,
                "url": params['img_url']
            },
            "gray_img": {
                "base64": img.to_base64(img.gray_img),
                "shape": img.gray_img.shape,
            },
            "filter2d_img": {
                "base64": img.to_base64(img.filter2d_img),
                "shape": img.filter2d_img.shape,
            },
            "threshold_img": {
                "base64": img.to_base64(img.threshold_img),
                "shape": img.threshold_img.shape
            },
            "gauss_img": {
                "base64": img.to_base64(img.gauss_img),
                "shape": img.gauss_img.shape
            },
            "canny_img": {
                "base64": img.to_base64(img.canny_img),
                "shape": img.canny_img.shape
            }
        })

