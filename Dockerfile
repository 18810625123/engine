FROM tensorflow/tensorflow:2.1.0rc2-py3

ARG git_revision_id
ENV APP_HOME=/home/app \
    GIT_REVISION_ID=${git_revision_id}

RUN groupadd -r app --gid=1000 \
 && useradd -r -m -g app -d $APP_HOME --uid=1000 app \
 && apt-get update \
 && apt-get install -y curl wget openssh-client gcc mysql-client libmysqlclient-dev libpq-dev sqlite3

RUN apt-get install -y --no-install-recommends bzr mercurial openssh-client subversion procps

RUN pip install django django-cors-headers djangorestframework mysqlclient libsvm lxml numpy numba pandas pillow matplotlib opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/

RUN apt-get install -y apt-file libsm6 libxrender1 autoconf automake bzip2 file g++ gcc imagemagick libbz2-dev libc6-dev libcurl4-openssl-dev libdb-dev libevent-dev libffi-dev libgdbm-dev libgeoip-dev libglib2.0-dev libjpeg-dev libkrb5-dev liblzma-dev libmagickcore-dev libmagickwand-dev libmysqlclient-dev libncurses-dev libpng-dev libpq-dev libreadline-dev libsqlite3-dev libssl-dev libtool libwebp-dev libxml2-dev libxslt-dev libyaml-dev make patch xz-utils zlib1g-dev

#RUN apt-get install -y --no-install-recommends bzr mercurial openssh-client subversion procps

WORKDIR $APP_HOME
RUN chown -R app:app $APP_HOME
COPY . $APP_HOME
RUN chown -R app:app $APP_HOME
USER app
EXPOSE 8002
CMD ["python", "manage.py", "runserver", "0.0.0.0:8002"]
