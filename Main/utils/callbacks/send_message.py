from pytorch_lightning.utilities import rank_zero_info
from utils.callbacks.base import BestEpochCallback
from pl_bolts.callbacks.printing import dicts_to_table
import smtplib
from email.mime.text import MIMEText
from email.header import Header

class SendMessageCallback(BestEpochCallback):
    def __init__(self, monitor='', mode='min'):
        super(SendMessageCallback, self).__init__(monitor=monitor, mode=mode)
        self.metrics_dict = {}
    

    def on_fit_end(self, trainer, pl_module):
        rank_zero_info(dicts_to_table([self.metrics_dict]))
        sender = 'xzalous@gmail.com'    
        receivers = ['1079891418@qq.com']
        message = MIMEText('test')
        message['From'] = Header("\\Omega")
        message['To'] =  Header("\\Alpha")
        message['Subject'] = Header('model train finished.\n')
        try:
            smtpObj = smtplib.SMTP('localhost')
            smtpObj.sendmail(sender, receivers, message.as_string())
        except smtplib.SMTPException:
            print ("Error: Unable to sned email")
