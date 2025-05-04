from django.db import models
from django.contrib.auth import get_user_model
import uuid

User = get_user_model()

class Company(models.Model):
    """
    Representa una empresa que puede tener varios códigos de invitación.
    """
    name = models.CharField("CQISA", max_length=200, unique=True)

    def __str__(self):
        return self.name

class InvitationCode(models.Model):
    """
    Código único asociado a una empresa. Se puede desactivar y se lleva un contador de usos.
    """
    code = models.CharField("Código", max_length=32, unique=True, default=uuid.uuid4().hex)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="invites")
    is_active = models.BooleanField("Activo", default=True)
    used_count = models.PositiveIntegerField("Veces usado", default=0)

    created_at = models.DateTimeField("Creado el", auto_now_add=True)

    def __str__(self):
        return f"{self.code} ({self.company.name})"

class Profile(models.Model):
    """
    Perfil extendido del usuario para guardar la empresa de invitación.
    Se crea tras el registro de forma automática (podrías usar señales para ello).
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    company = models.ForeignKey(Company, null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        return f"Perfil de {self.user.username}"
