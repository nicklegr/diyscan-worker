FROM ubuntu:20.04

ENV TZ Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev libleptonica-dev pkg-config

RUN apt-get install -y python3 python3-pip

# RUN apt-get -y clean
# RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python tesserocr pyocr
RUN apt-get install -y curl

RUN apt-get install -y libsm6 libxrender1 libxext-dev

RUN curl -L -o /usr/share/tesseract-ocr/4.00/tessdata/jpn.traineddata https://github.com/tesseract-ocr/tessdata/raw/4.0.0/jpn.traineddata

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# スクリプトに変更があっても、bundle installをキャッシュさせる
# COPY Gemfile /usr/src/app/
# COPY Gemfile.lock /usr/src/app/
# RUN bundle install --deployment --without=test --jobs 4

COPY . /usr/src/app

EXPOSE 80
