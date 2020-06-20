FROM ubuntu:20.04

RUN sed -i.bak -e "s%http://[^ ]\+%http://ftp.iij.ad.jp/pub/linux/ubuntu/archive/%g" /etc/apt/sources.list

ENV TZ Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
  apt-get install -y \
    tesseract-ocr libtesseract-dev libleptonica-dev pkg-config \
    python3 python3-pip python3-dev \
    curl \
    libsm6 libxrender1 libxext-dev

# RUN apt-get -y clean
# RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python tesserocr

RUN curl -L -o /usr/share/tesseract-ocr/4.00/tessdata/jpn.traineddata https://github.com/tesseract-ocr/tessdata/raw/4.0.0/jpn.traineddata

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# スクリプトに変更があっても、bundle installをキャッシュさせる
# COPY Gemfile /usr/src/app/
# COPY Gemfile.lock /usr/src/app/
# RUN bundle install --deployment --without=test --jobs 4

COPY . /usr/src/app

EXPOSE 80
