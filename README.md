# anomaly_detection-using-autoEncoder
Anomaly detection in a video using Auto Encoder

The main idea is to assess the quality of video. While playing video in a test machine if corruptions like flickers, green corruptions, black screen .etc occurs, this model can detect them and define as BAD quality.

The inference dataset shoule be training dataset. As model tries to replicate the trained images, it can accurately encode && decode them and result very less reconsturction loss.
But if some new image which is not known to model, this conversion will not be correct. Hence it will give more loss.

By defining the right threshold, we can able to identify both the flows.
