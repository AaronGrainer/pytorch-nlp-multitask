include .env

download-coco:
	mkdir -p data/raw/coco
	mkdir -p data/tfrecord/coco
	cd data/raw/coco; wget http://images.cocodataset.org/zips/train2017.zip
	cd data/raw/coco; wget http://images.cocodataset.org/zips/val2017.zip
	cd data/raw/coco; wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

	cd data/raw/coco; unzip train2017.zip
	cd data/raw/coco; unzip val2017.zip
	cd data/raw/coco; unzip annotations_trainval2017.zip

create-coco-tfrecord:
	cd data; python -m create_coco_tfrecord

run-local:
	python -m beam_dag_runner

create-service-account:
	gcloud iam service-accounts create ${SERVICE_ACCOUNT} \
		--display-name=${SERVICE_ACCOUNT} \
		--project=${GCP_PROJECT_ID}
	gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
		--member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
		--role=roles/logging.logWriter
	gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
		--member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
		--role=roles/monitoring.metricWriter
	gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
		--member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
		--role=roles/monitoring.viewer
	gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
		--member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
		--role=roles/storage.admin
	gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
		--member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
		--role=roles/bigquery.user
	gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
		--member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
		--role=roles/dataflow.admin
	gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
		--member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
		--role=roles/ml.developer

create-cluster:
	gcloud container clusters create ${CLUSTER_NAME} \
		--machine-type n1-standard-2 \
		--num-nodes 3 \
		--service-account ${SERVICE_ACCOUNT}@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
		--preemptible

install-skaffold:
	curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
	sudo install skaffold /usr/local/bin/

tfx-create-pipeline:
	tfx pipeline create \
		--pipeline-path=kubeflow_dag_runner.py \
		--endpoint=${ENDPOINT} \
		--build-target-image=${CUSTOM_TFX_IMAGE}

tfx-update-pipeline:
	tfx pipeline update \
		--pipeline-path=kubeflow_dag_runner.py \
		--endpoint=${ENDPOINT}

tfx-delete-pipeline:
	tfx pipeline delete \
		--pipeline-name=${PIPELINE_NAME} \
		--endpoint=${ENDPOINT}

tfx-run:
	tfx run create \
		--pipeline-name=${PIPELINE_NAME} \
		--endpoint=${ENDPOINT}

create-service-account-key:
	gcloud iam service-accounts keys create key.json \
		--iam-account ${SERVICE_ACCOUNT_GITHUB}