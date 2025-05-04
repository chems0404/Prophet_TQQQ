from django.contrib import admin
from .models import Company, InvitationCode, Profile

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)

@admin.register(InvitationCode)
class InvitationCodeAdmin(admin.ModelAdmin):
    list_display = ('code', 'company', 'used_count', 'is_active', 'created_at')
    list_filter  = ('company', 'is_active')
    search_fields = ('code', 'company__name')

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'company')
    search_fields  = ('user__username', 'company__name')
