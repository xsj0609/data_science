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
        self.instance_type = 'ecs.t5-c1m1.large'
        # 实例的计费方式
        self.instance_charge_type = 'PostPaid'
        # 镜像ID
        self.image_id = 'ubuntu_16_04_64_20G_alibase_20190301.vhd'
        # 购买资源的时长
        self.period = 1
        # 购买资源的时长单位
        self.period_unit = 'Hourly'
        # 实例所属的可用区编号
        self.zone_id = 'random'
        # 网络计费类型
        self.internet_charge_type = 'PayByTraffic'
        # 实例名称
        self.instance_name = 'launch-advisor-20190413'
        # 指定创建ECS实例的数量
        self.amount = 1
        # 公网出带宽最大值
        self.internet_max_bandwidth_out = 5
        # 是否为I/O优化实例
        self.io_optimized = 'optimized'
        # 密钥对名称
        self.key_pair_name = 'aliyun-key'
        # 是否开启安全加固
        self.security_enhancement_strategy = 'Active'
        # 系统盘大小
        self.system_disk_size = '40'
        # 系统盘的磁盘种类
        self.system_disk_category = 'cloud_efficiency'
        # 数据盘
        self.data_disks = [
            {
               'Size': 20,
               'Category': 'cloud_efficiency',
               'Encrypted': False,
               'DeleteWithInstance': True
            }
        ]
        
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
        request.set_Period(self.period)
        request.set_PeriodUnit(self.period_unit)
        request.set_ZoneId(self.zone_id)
        request.set_InternetChargeType(self.internet_charge_type)
        request.set_InstanceName(self.instance_name)
        request.set_Amount(self.amount)
        request.set_InternetMaxBandwidthOut(self.internet_max_bandwidth_out)
        request.set_IoOptimized(self.io_optimized)
        request.set_KeyPairName(self.key_pair_name)
        request.set_SecurityEnhancementStrategy(self.security_enhancement_strategy)
        request.set_SystemDiskSize(self.system_disk_size)
        request.set_SystemDiskCategory(self.system_disk_category)
        request.set_DataDisks(self.data_disks)
         
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