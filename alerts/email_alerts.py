import smtplib, yaml

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

def send_alert(aqi):
    if aqi < config["alerts"]["aqi_threshold"]:
        return

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(
        config["alerts"]["sender_email"],
        config["alerts"]["sender_password"]
    )
    server.sendmail(
        config["alerts"]["sender_email"],
        config["alerts"]["receiver_email"],
        f"⚠ AQI ALERT: AQI reached {aqi}"
    )
    server.quit()