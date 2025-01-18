import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

class EmailNotification:
    def __init__(self, from_email, from_password, smtp_server='smtp.gmail.com', smtp_port=587):
        """
        初始化 EmailNotification 類別

        Args:
            from_email (str): 發件人電子郵件地址
            from_password (str): 發件人電子郵件密碼或應用程式密碼
            smtp_server (str): SMTP 伺服器，預設為 Gmail 的 SMTP
            smtp_port (int): SMTP 埠號，預設為 Gmail 的 587
        """
        self.from_email = from_email
        self.from_password = from_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send_email(self, to_email, subject, body):
        """
        發送電子郵件

        Args:
            to_email (str): 收件人電子郵件地址
            subject (str): 郵件主題
            body (str): 郵件內容
        """

        # 如果 to_email 是單個地址，轉換為列表格式
        if isinstance(to_email, str):
            to_email = [to_email]

        # 將收件人地址列表拼接成逗號分隔的字串
        to_email = ', '.join(to_email)

        # 設置郵件內容
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # 發送郵件
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_email, self.from_password)
                server.send_message(msg)
            logging.info("Email notification sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send email notification. Error: {e}")
