from django import forms


class NdForm(forms.Form):
    nd_h = forms.FloatField(label='h')
    nd_k = forms.FloatField(label='k')
    nd_l = forms.FloatField(label='l')


class RdForm(forms.Form):
    rd_h = forms.FloatField(label="h")
    rd_k = forms.FloatField(label="k")
    rd_l = forms.FloatField(label="l")


class NdForm1(forms.Form):
    nd_h1 = forms.FloatField(label='h')
    nd_k1 = forms.FloatField(label='k')
    nd_l1 = forms.FloatField(label='l')


class RdForm1(forms.Form):
    rd_h1 = forms.FloatField(label="h")
    rd_k1 = forms.FloatField(label="k")
    rd_l1 = forms.FloatField(label="l")


class NdForm2(forms.Form):
    nd_h2 = forms.FloatField(label='h')
    nd_k2 = forms.FloatField(label='k')
    nd_l2 = forms.FloatField(label='l')


class RdForm2(forms.Form):
    rd_h2 = forms.FloatField(label="h")
    rd_k2 = forms.FloatField(label="k")
    rd_l2 = forms.FloatField(label="l")


class PhiForm(forms.Form):
    phi = forms.FloatField(label='')


class ThetaForm(forms.Form):
    theta = forms.FloatField(label='')


class RotationForm(forms.Form):
    rot = forms.FloatField(label='')


class DirectionForm1(forms.Form):
    d1_h = forms.FloatField(label='h')
    d1_k = forms.FloatField(label='k')
    d1_l = forms.FloatField(label='l')


class DirectionForm2(forms.Form):
    d2_h = forms.FloatField(label='h')
    d2_k = forms.FloatField(label='k')
    d2_l = forms.FloatField(label='l')


class PlaneForm1(forms.Form):
    p1_h = forms.FloatField(label='h')
    p1_k = forms.FloatField(label='k')
    p1_l = forms.FloatField(label='l')


class PlaneForm2(forms.Form):
    p2_h = forms.FloatField(label='h')
    p2_k = forms.FloatField(label='k')
    p2_l = forms.FloatField(label='l')
