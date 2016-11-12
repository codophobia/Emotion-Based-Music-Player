# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2016-11-09 15:26
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('music', '0006_auto_20161103_1533'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='song',
            name='song_logo',
        ),
        migrations.RemoveField(
            model_name='song',
            name='song_type',
        ),
        migrations.AddField(
            model_name='song',
            name='song_singer',
            field=models.CharField(default='unknown artist', max_length=250),
        ),
    ]