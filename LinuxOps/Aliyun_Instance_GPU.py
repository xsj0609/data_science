#!/usr/bin/env python
# coding=utf-8
import json
import time
import traceback

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
from aliyunsdkecs.request.v20140526.RunInstancesRequest import RunInstancesRequest
from aliyunsdkecs.request.v20140526.DescribeInstancesRequest import DescribeInstancesRequest


RUNNING_STATUS = 'Running'
CHECK_INTERVAL = 3
CHECK_TIMEOUT = 180


class AliyunRunInstancesExample(object):

    def __init__(self):
        self.access_id = '<AccessKey>'
        self.access_secret = '<AccessSecret>'

        # 是否只预检此次请求。true：发送检查请求，不会创建实例，也不会产生费用；false：发送正常请求，通过检查后直接创建实例，并直接产生费用
        self.dry_run = False
        # 实例所属的地域ID
        self.region_id = 'cn-shanghai'
        # 实例的资源规格
        self.instance_type = 'ecs.gn5i-c8g1.2xlarge'
        # 实例的计费方式
        self.instance_charge_type = 'PostPaid'
        # 镜像ID
        self.image_id = 'ubuntu_16_04_64_20G_alibase_20190301.vhd'
        # 指定新创建实例所属于的安全组ID
        self.security_group_id = 'sg-uf67sfx4012ehtzgul3m'
        # 购买资源的时长
        self.period = 1
        # 购买资源的时长单位
        self.period_unit = 'Hourly'
        # 实例所属的可用区编号
        self.zone_id = 'random'
        # 网络计费类型
        self.internet_charge_type = 'PayByTraffic'
        # 虚拟交换机ID
        self.vswitch_id = 'vsw-uf6i0g5mf2c5bvo8ujcah'
        # 实例名称
        self.instance_name = 'launch-advisor-20190415'
        # 指定创建ECS实例的数量
        self.amount = 1
        # 公网出带宽最大值
        self.internet_max_bandwidth_out = 5
        # 云服务器的主机名
        self.host_name = 'xusj'
        # 是否为I/O优化实例
        self.io_optimized = 'optimized'
        # 实例自定义数据
        self.user_data = 'IyEvYmluL3NoCgpsb2c9Ii9yb290L252aWRpYV9pbnN0YWxsLmxvZyIKCmRyaXZlcl92ZXJzaW9uPSIzOTAuNDYiCmN1ZGFfdmVyc2lvbj0iOS4wLjE3NiIKCmN1ZGFfYmlnX3ZlcnNpb249JChlY2hvICRjdWRhX3ZlcnNpb24gfCBhd2sgLUYnLicgJ3twcmludCAkMSIuIiQyfScpCgplY2hvICJpbnN0YWxsIG52aWRpYSBkcml2ZXIgYW5kIGN1ZGEgYmVnaW4gLi4uLi4uIiA+PiAkbG9nIDI+JjEKZWNobyAiZHJpdmVyIHZlcnNpb246ICRkcml2ZXJfdmVyc2lvbiIgPj4gJGxvZyAyPiYxCmVjaG8gImN1ZGEgdmVyc2lvbjogJGN1ZGFfdmVyc2lvbiIgPj4gJGxvZyAyPiYxCgojIyMjIyNVYnVudHUjIyMjIyMjIyMjCmNyZWF0ZV9udmlkaWFfcmVwb191YnVudHUoKQp7CiAgICBpZiBbIC1mICIvZXRjL2FwdC9zb3VyY2VzLmxpc3QuZC9zb3VyY2VzLWFsaXl1bi0wLmxpc3QiIF07IHRoZW4KICAgICAgICByZXBvX2ZpbGU9Ii9ldGMvYXB0L3NvdXJjZXMubGlzdC5kL3NvdXJjZXMtYWxpeXVuLTAubGlzdCIKICAgIGVsc2UKICAgICAgICByZXBvX2ZpbGU9Ii9ldGMvYXB0L3NvdXJjZXMubGlzdCIKICAgIGZpCgogICAgdXJsPSQoY2F0ICRyZXBvX2ZpbGUgfGdyZXAgIl5kZWIiIHwgaGVhZCAtMSB8IGF3ayAtRidbL10nICd7cHJpbnQgJDEiLy8iJDN9JykKICAgIGlmIFsgLXogIiR1cmwiIF07IHRoZW4KICAgICAgICB1cmw9Imh0dHA6Ly9taXJyb3JzLmNsb3VkLmFsaXl1bmNzLmNvbSIKICAgIGZpCgogICAgdXJsMT0iJHVybC9vcHN4L2Vjcy9saW51eC9hcHQvIGVjcyBjdWRhIgogICAgdXJsMj0iJHVybC9vcHN4L2Vjcy9saW51eC9hcHQvIGVjcyBkcml2ZXIiCiAgICBlY2hvICR1cmwxID4gL2V0Yy9hcHQvc291cmNlcy5saXN0LmQvbnZpZGlhLmxpc3QKICAgIGVjaG8gJHVybDIgPj4gL2V0Yy9hcHQvc291cmNlcy5saXN0LmQvbnZpZGlhLmxpc3QKCiAgICB3Z2V0IC1PIC0gJHVybC9vcHN4L29wc3hAc2VydmljZS5hbGliYWJhLmNvbS5ncGcua2V5IHwgYXB0LWtleSBhZGQgLQogICAgYXB0IHVwZGF0ZSA+PiAkbG9nIDI+JjEKCn0KCgppbnN0YWxsX2tlcm5lbF91YnVudHUoKQp7CiAgICAjaW5zdGFsbCBsaW51eC1oZWFkZXJzCiAgICBrZXJuZWxfdmVyc2lvbj0kKHVuYW1lIC1yKQogICAgZWNobyAiKioqKioqZXhlYyBcInVuYW1lIC1yXCI6ICRrZXJuZWxfdmVyc2lvbiIKICAgIGVjaG8gIioqKioqKmV4ZWMgXCJkcGtnIC0tbGlzdCB8Z3JlcCBsaW51eC1oZWFkZXJzIHwgZ3JlcCAka2VybmVsX3ZlcnNpb24gfCB3YyAtbFwiIgogICAgbGludXhfaGVhZGVyc19udW09JChkcGtnIC0tbGlzdCB8Z3JlcCBsaW51eC1oZWFkZXJzIHwgZ3JlcCAka2VybmVsX3ZlcnNpb24gfCB3YyAtbCkKICAgIGVjaG8gIioqKioqKmxpbnV4X2hlYWRlcnNfbnVtPSRsaW51eF9oZWFkZXJzX251bSIKICAgIGlmIFsgJGxpbnV4X2hlYWRlcnNfbnVtIC1lcSAwIF07dGhlbgogICAgICAgIGVjaG8gIioqKioqKmV4ZWMgXCJhcHQtZ2V0IGluc3RhbGwgLXkgLS1hbGxvdy11bmF1dGhlbnRpY2F0ZWQgbGludXgtaGVhZGVycy0ka2VybmVsX3ZlcnNpb25cIiIKICAgICAgICBhcHQtZ2V0IGluc3RhbGwgLXkgLS1hbGxvdy11bmF1dGhlbnRpY2F0ZWQgbGludXgtaGVhZGVycy0ka2VybmVsX3ZlcnNpb24KICAgICAgICBpZiBbICQ/IC1uZSAwIF07IHRoZW4KICAgICAgICAgICAgZWNobyAiZXJyb3I6IGluc3RhbGwgbGludXgtaGVhZGVycyBmYWlsISEhIgogICAgICAgICAgICByZXR1cm4gMQogICAgICAgIGZpCiAgICBmaQp9CgppbnN0YWxsX2RyaXZlcl91YnVudHUoKQp7CiAgICAjaW5zdGFsbCBkcml2ZXIKICAgIGRyaXZlcl9maWxlX251bT0kKGFwdC1jYWNoZSBzZWFyY2ggbnZpZGlhIHwgZ3JlcCBkcml2ZXIgfCBncmVwICRyZWxlYXNlIHwgZ3JlcCAkZHJpdmVyX3ZlcnNpb24gfCB3YyAtbCkKICAgIGlmIFsgJGRyaXZlcl9maWxlX251bSAtZXEgMSBdO3RoZW4KICAgICAgICBkcml2ZXJfZmlsZT0kKGFwdC1jYWNoZSBzZWFyY2ggbnZpZGlhIHwgZ3JlcCBkcml2ZXIgfCBncmVwICRyZWxlYXNlIHwgZ3JlcCAkZHJpdmVyX3ZlcnNpb24gfCBhd2sgLUYnICcgJ3twcmludCAkMX0nKQogICAgICAgIGVjaG8gIioqKioqKmV4ZWMgXCJhcHQtY2FjaGUgc2VhcmNoIG52aWRpYSB8IGdyZXAgZHJpdmVyIHxncmVwICRyZWxlYXNlIHxncmVwICRkcml2ZXJfdmVyc2lvbiB8IGF3ayAtRicgJyAne3ByaW50IFwkMX0nXCI6IgogICAgICAgIGVjaG8gJGRyaXZlcl9maWxlCiAgICBlbHNlCiAgICAgICAgZWNobyAiZXJyb3I6IGRyaXZlcl9maWxlX251bSA9ICRkcml2ZXJfZmlsZV9udW0gLCBnZXQgZHJpdmVyIGZpbGUgZmFpbGVkLCBleGl0IgogICAgICAgIHJldHVybiAxCiAgICBmaQoKICAgIGVjaG8gIioqKioqKmV4ZWMgXCJhcHQtZ2V0IGluc3RhbGwgLXkgLS1hbGxvdy11bmF1dGhlbnRpY2F0ZWQgJGRyaXZlcl9maWxlXCIgIgogICAgYXB0LWdldCBpbnN0YWxsIC15IC0tYWxsb3ctdW5hdXRoZW50aWNhdGVkICRkcml2ZXJfZmlsZQoKICAgIGVjaG8gIioqKioqKmV4ZWMgXCJhcHQta2V5IGFkZCAvdmFyL252aWRpYSpkcml2ZXIqJGRyaXZlcl92ZXJzaW9uLyoucHViXCIiCiAgICBhcHQta2V5IGFkZCAvdmFyL252aWRpYSpkcml2ZXIqJGRyaXZlcl92ZXJzaW9uLyoucHViCgogICAgZWNobyAiKioqKioqZXhlYyBcImFwdC1nZXQgdXBkYXRlICYmIGFwdC1nZXQgaW5zdGFsbCAteSAtLWFsbG93LXVuYXV0aGVudGljYXRlZCBjdWRhLWRyaXZlcnNcIiAiCiAgICBhcHQtZ2V0IHVwZGF0ZSAmJiBhcHQtZ2V0IGluc3RhbGwgLXkgLS1hbGxvdy11bmF1dGhlbnRpY2F0ZWQgY3VkYS1kcml2ZXJzCgogICAgaWYgWyAkPyAtbmUgMCBdOyB0aGVuCiAgICAgICAgZWNobyAiZXJyb3I6IGRyaXZlciBpbnN0YWxsIGZhaWwhISEiCiAgICAgICAgcmV0dXJuIDEKICAgIGZpCn0KCgppbnN0YWxsX2N1ZGFfdWJ1bnR1KCkKewogICAgYmVnaW5fY3VkYT0kKGRhdGUgJyslcycpCiAgICBjdWRhX2ZpbGVfbnVtPSQoYXB0LWNhY2hlIHNlYXJjaCBjdWRhIHwgZ3JlcCAkcmVsZWFzZSB8IGdyZXAgJGN1ZGFfYmlnX3ZlcnNpb24gfGdyZXAgLXYgdXBkYXRlIHwgd2MgLWwpCiAgICBpZiBbICRjdWRhX2ZpbGVfbnVtIC1lcSAxIF07dGhlbgogICAgICAgIGN1ZGFfZmlsZT0kKGFwdC1jYWNoZSBzZWFyY2ggY3VkYSB8IGdyZXAgJHJlbGVhc2UgfCBncmVwICRjdWRhX2JpZ192ZXJzaW9uIHxncmVwIC12IHVwZGF0ZSB8IGF3ayAtRicgJyAne3ByaW50ICQxfScpCiAgICAgICAgZWNobyAiKioqKioqZXhlYyBcImFwdC1jYWNoZSBzZWFyY2ggY3VkYXwgZ3JlcCAkcmVsZWFzZXwgZ3JlcCAkY3VkYV9iaWdfdmVyc2lvbiB8Z3JlcCAtdiB1cGRhdGV8IGF3ayAtRicgJyAne3ByaW50IFwkMX0nXCIiCiAgICAgICAgZWNobyAkY3VkYV9maWxlCiAgICBlbHNlCiAgICAgICAgZWNobyAiZXJyb3I6IGN1ZGFfZmlsZV9udW0gPSAkY3VkYV9maWxlX251bSAsIGdldCBjdWRhIGZpbGUgZmFpbGVkLCBleGl0IgogICAgICAgIHJldHVybiAxCiAgICBmaQoKICAgICNpbnN0YWxsIGN1ZGEKICAgIGVjaG8gIioqKioqKmV4ZWMgXCJhcHQtZ2V0IGluc3RhbGwgLXkgLS1hbGxvdy11bmF1dGhlbnRpY2F0ZWQgJGN1ZGFfZmlsZVwiICIKICAgIGFwdC1nZXQgaW5zdGFsbCAteSAtLWFsbG93LXVuYXV0aGVudGljYXRlZCAkY3VkYV9maWxlCgogICAgZW5kX2N1ZGFfdW5wYWNrPSQoZGF0ZSAnKyVzJykKICAgIHRpbWVfY3VkYV91bnBhY2s9JCgoZW5kX2N1ZGFfdW5wYWNrLWJlZ2luX2N1ZGEpKQogICAgZWNobyAiKioqKioqZG93bmxvYWQgYW5kIHVucGFjayBjdWRhIGZpbGUgZW5kLCBlbmQgdGltZTogJGVuZF9jdWRhX3VucGFjaywgdXNlIHRpbWUgJHRpbWVfY3VkYV91bnBhY2sgcyIKCiAgICBlY2hvICIqKioqKipleGVjIFwiYXB0LWNhY2hlIHNlYXJjaCBjdWRhIHwgZ3JlcCAkcmVsZWFzZSB8IGdyZXAgJGN1ZGFfYmlnX3ZlcnNpb24gfCBncmVwIHVwZGF0ZSB8IGF3ayAtRicgJyAne3ByaW50IFwkMX0nXCIgIgogICAgY3VkYV9wYXRjaF9maWxlbGlzdD0kKGFwdC1jYWNoZSBzZWFyY2ggY3VkYSB8IGdyZXAgJHJlbGVhc2UgfCBncmVwICRjdWRhX2JpZ192ZXJzaW9uIHwgZ3JlcCB1cGRhdGUgfCBhd2sgLUYnICcgJ3twcmludCAkMX0nKQoKICAgIGVjaG8gIioqKioqKiBjdWRhX3BhdGNoX2ZpbGVsaXN0IgogICAgZWNobyAkY3VkYV9wYXRjaF9maWxlbGlzdAogICAgZm9yIGN1ZGFfcGF0Y2hfZmlsZSBpbiAkY3VkYV9wYXRjaF9maWxlbGlzdAogICAgZG8KICAgICAgICBlY2hvICIqKioqKipleGVjIFwiYXB0LWdldCBpbnN0YWxsIC15IC0tYWxsb3ctdW5hdXRoZW50aWNhdGVkICRjdWRhX3BhdGNoX2ZpbGVcIiAiCiAgICAgICAgYXB0LWdldCBpbnN0YWxsIC15IC0tYWxsb3ctdW5hdXRoZW50aWNhdGVkICRjdWRhX3BhdGNoX2ZpbGUKCiAgICBkb25lCgogICAgZWNobyAiKioqKioqZXhlYyBcImFwdC1nZXQgdXBkYXRlICYmIGFwdC1nZXQgaW5zdGFsbCAteSAtLWFsbG93LXVuYXV0aGVudGljYXRlZCBjdWRhXCIgIgogICAgYXB0LWdldCB1cGRhdGUgJiYgYXB0LWdldCBpbnN0YWxsIC15IC0tYWxsb3ctdW5hdXRoZW50aWNhdGVkIGN1ZGEKICAgIGlmIFsgJD8gLW5lIDAgXTsgdGhlbgogICAgICAgIGVjaG8gImVycm9yOiBjdWRhIGluc3RhbGwgZmFpbCEhISIKICAgICAgICByZXR1cm4gMQogICAgZmkKCiAgICBlbmRfY3VkYT0kKGRhdGUgJyslcycpCiAgICB0aW1lX2N1ZGE9JCgoZW5kX2N1ZGEtYmVnaW5fY3VkYSkpCiAgICBlY2hvICIqKioqKippbnN0YWxsIGN1ZGEgYmVnaW4gdGltZTogJGJlZ2luX2N1ZGEsIGVuZCB0aW1lICRlbmRfY3VkYSwgdXNlIHRpbWUgJHRpbWVfY3VkYSBzIgp9CgplbmFibGVfcG0oKQp7CiAgICBlY2hvICIjIS9iaW4vYmFzaCIgfCB0ZWUgLWEgL2V0Yy9pbml0LmQvZW5hYmxlX3BtLnNoCiAgICBlY2hvICJudmlkaWEtc21pIC1wbSAxIiB8IHRlZSAtYSAvZXRjL2luaXQuZC9lbmFibGVfcG0uc2gKICAgIGVjaG8gImV4aXQgMCIgfCB0ZWUgLWEgL2V0Yy9pbml0LmQvZW5hYmxlX3BtLnNoCgogICAgY2htb2QgK3ggL2V0Yy9pbml0LmQvZW5hYmxlX3BtLnNoCgogICAgc3RyPSQoY2F0ICRmaWxlbmFtZSB8Z3JlcCAiZXhpdCIpCiAgICBpZiBbIC16ICIkc3RyIiBdOyB0aGVuCiAgICAgICAgZWNobyAiL2V0Yy9pbml0LmQvZW5hYmxlX3BtLnNoIiB8IHRlZSAtYSAkZmlsZW5hbWUKICAgIGVsc2UKICAgICAgICBzZWQgLWkgJy9leGl0L2lcL2V0Yy9pbml0LmQvZW5hYmxlX3BtLnNoJyAkZmlsZW5hbWUKICAgIGZpCiAgICBjaG1vZCAreCAkZmlsZW5hbWUKCn0KCmlmIFsgISAtZiAiL3Vzci9iaW4vbHNiX3JlbGVhc2UiIF07IHRoZW4KICAgIGFwdC1nZXQgaW5zdGFsbCAteSBsc2ItcmVsZWFzZQpmaQoKc3RyPSQobHNiX3JlbGVhc2UgLWkgfCBhd2sgLUYnOicgJ3twcmludCAkMn0nKQpvcz0kKGVjaG8gJHN0ciB8IHNlZCAncy8gLy9nJykKaWYgWyAiJG9zIiA9ICJVYnVudHUiIF07IHRoZW4KICAgIG9zPSJ1YnVudHUiCiAgICBzdHI9JChsc2JfcmVsZWFzZSAtciB8IGF3ayAtRidbOi5dJyAne3ByaW50ICQyJDN9JykKICAgIHZlcnNpb249JChlY2hvICRzdHIgfCBzZWQgJ3MvIC8vZycpCiAgICByZWxlYXNlPSJ1YnVudHUke3ZlcnNpb259IgogICAgZmlsZW5hbWU9Ii9ldGMvcmMubG9jYWwiCmVsc2UKICAgIGVjaG8gIkVSUk9SOiBPUyAoJG9zKSBpcyBpbnZhbGlkISIgPj4gJGxvZyAyPiYxCiAgICBleGl0IDEKZmkKCmVjaG8gIm9zOiRvcyByZWxlYXNlOiRyZWxlYXNlIHZlcnNpb246JHZlcnNpb24iID4+ICRsb2cgMj4mMQoKY3JlYXRlX252aWRpYV9yZXBvX3VidW50dSAKCmJlZ2luPSQoZGF0ZSAnKyVzJykKaW5zdGFsbF9rZXJuZWxfdWJ1bnR1ID4+ICRsb2cgMj4mMSAKaWYgWyAkPyAtbmUgMCBdOyB0aGVuCiAgICBlY2hvICJlcnJvcjogIGtlcm5lbCBpbnN0YWxsIGZhaWwhISEiID4+ICRsb2cgMj4mMQogICAgZXhpdCAxCmZpCgplbmQ9JChkYXRlICcrJXMnKQp0aW1lX2tlcm5lbD0kKChlbmQtYmVnaW4pKQplY2hvICIqKioqKippbnN0YWxsIGtlcm5lbC1kZXZlbCBiZWdpbiB0aW1lOiAkYmVnaW4sIGVuZCB0aW1lOiAkZW5kLCB1c2UgdGltZTogJHRpbWVfa2VybmVsIHMiID4+ICRsb2cgMj4mMQoKCmJlZ2luX2RyaXZlcj0kKGRhdGUgJyslcycpCmluc3RhbGxfZHJpdmVyX3VidW50dSA+PiAkbG9nIDI+JjEgCmlmIFsgJD8gLW5lIDAgXTsgdGhlbgogICAgZWNobyAiZXJyb3I6ICBkcml2ZXIgaW5zdGFsbCBmYWlsISEhIiA+PiAkbG9nIDI+JjEKICAgIGV4aXQgMQpmaQoKZW5kX2RyaXZlcj0kKGRhdGUgJyslcycpCnRpbWVfZHJpdmVyPSQoKGVuZF9kcml2ZXItYmVnaW5fZHJpdmVyKSkKZWNobyAiKioqKioqaW5zdGFsbCBkcml2ZXIgYmVnaW4gdGltZTogJGJlZ2luX2RyaXZlciwgZW5kIHRpbWU6ICRlbmRfZHJpdmVyLCAgdXNlIHRpbWU6ICR0aW1lX2RyaXZlciBzIiA+PiAkbG9nIDI+JjEKCmluc3RhbGxfY3VkYV91YnVudHUgPj4gJGxvZyAyPiYxIAppZiBbICQ/IC1uZSAwIF07IHRoZW4KICAgIGVjaG8gImVycm9yOiAgY3VkYSBpbnN0YWxsIGZhaWwhISEiID4+ICRsb2cgMj4mMQogICAgZXhpdCAxCmZpCgoKZWNobyAiKioqKioqaW5zdGFsbCBrZXJuZWwtZGV2ZWwgdXNlIHRpbWUgJHRpbWVfa2VybmVsIHMiID4+ICRsb2cgMj4mMQplY2hvICIqKioqKippbnN0YWxsIGRyaXZlciB1c2UgdGltZSAkdGltZV9kcml2ZXIgcyIgPj4gJGxvZyAyPiYxCmVjaG8gIioqKioqKmluc3RhbGwgY3VkYSB1c2UgdGltZSAkdGltZV9jdWRhIHMiID4+ICRsb2cgMj4mMQoKZWNobyAiYWRkIGF1dG8gZW5hYmxlIFBlcnNpc3RlbmNlIE1vZGUgd2hlbiBzdGFydCB2bS4uLiIgPj4gJGxvZyAyPiYxCmVuYWJsZV9wbQoKZWNobyAicmVib290Li4uLi4uIiA+PiAkbG9nIDI+JjEKcmVib290'
        # 密钥对名称
        self.key_pair_name = 'aliyun-key'
        # 是否开启安全加固
        self.security_enhancement_strategy = 'Active'
        # 系统盘大小
        self.system_disk_size = '40'
        # 系统盘的磁盘种类
        self.system_disk_category = 'cloud_efficiency'
        
        self.client = AcsClient(self.access_id, self.access_secret, self.region_id)

    def run(self):
        try:
            ids = self.run_instances()
            self._check_instances_status(ids)
        except ClientException as e:
            print('Fail. Something with your connection with Aliyun go incorrect.'
                  ' Code: {code}, Message: {msg}'
                  .format(code=e.error_code, msg=e.message))
        except ServerException as e:
            print('Fail. Business error.'
                  ' Code: {code}, Message: {msg}'
                  .format(code=e.error_code, msg=e.message))
        except Exception:
            print('Unhandled error')
            print(traceback.format_exc())

    def run_instances(self):
        """
        调用创建实例的API，得到实例ID后继续查询实例状态
        :return:instance_ids 需要检查的实例ID
        """
        request = RunInstancesRequest()
       
        request.set_DryRun(self.dry_run)
        
        request.set_InstanceType(self.instance_type)
        request.set_InstanceChargeType(self.instance_charge_type)
        request.set_ImageId(self.image_id)
        request.set_SecurityGroupId(self.security_group_id)
        request.set_Period(self.period)
        request.set_PeriodUnit(self.period_unit)
        request.set_ZoneId(self.zone_id)
        request.set_InternetChargeType(self.internet_charge_type)
        request.set_VSwitchId(self.vswitch_id)
        request.set_InstanceName(self.instance_name)
        request.set_Amount(self.amount)
        request.set_InternetMaxBandwidthOut(self.internet_max_bandwidth_out)
        request.set_HostName(self.host_name)
        request.set_IoOptimized(self.io_optimized)
        request.set_UserData(self.user_data)
        request.set_KeyPairName(self.key_pair_name)
        request.set_SecurityEnhancementStrategy(self.security_enhancement_strategy)
        request.set_SystemDiskSize(self.system_disk_size)
        request.set_SystemDiskCategory(self.system_disk_category)
         
        body = self.client.do_action_with_exception(request)
        data = json.loads(body)
        instance_ids = data['InstanceIdSets']['InstanceIdSet']
        print('Success. Instance creation succeed. InstanceIds: {}'.format(', '.join(instance_ids)))
        return instance_ids

    def _check_instances_status(self, instance_ids):
        """
        每3秒中检查一次实例的状态，超时时间设为3分钟.
        :param instance_ids 需要检查的实例ID
        :return:
        """
        start = time.time()
        while True:
            request = DescribeInstancesRequest()
            request.set_InstanceIds(json.dumps(instance_ids))
            body = self.client.do_action_with_exception(request)
            data = json.loads(body)
            for instance in data['Instances']['Instance']:
                if RUNNING_STATUS in instance['Status']:
                    instance_ids.remove(instance['InstanceId'])
                    print('Instance boot successfully: {}'.format(instance['InstanceId']))

            if not instance_ids:
                print('Instances all boot successfully')
                break

            if time.time() - start > CHECK_TIMEOUT:
                print('Instances boot failed within {timeout}s: {ids}'
                      .format(timeout=CHECK_TIMEOUT, ids=', '.join(instance_ids)))
                break

            time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    AliyunRunInstancesExample().run()