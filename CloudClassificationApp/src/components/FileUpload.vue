<template>
  <v-container>
    <p class="mt-8">Sube las imágenes de nubes que deseas clasificar:</p>
    <v-form ref="form" @submit.prevent="handleSubmit">
      <v-file-input
        chips
        multiple
        label="Click aquí para seleccionar las imágenes a subir"
        show-size
        counter
        clearable
        clear-icon="mdi-close-circle"
        v-model="files"
        accept="image/png, image/jpeg, image/jpg"
        :disabled="files.length > 0 && formSubmit"
      >
      </v-file-input>
      <v-btn
        class="mt-3"
        color="primary"
        v-if="!formSubmit"
        type="submit"
        :class="{ 'disable-events': files.length === 0 }"
      >
        Generar predicción</v-btn
      >
    </v-form>
    <div
      class="mt-3 d-flex align-center"
      v-if="progressStatus > 0 && progressStatus < 100"
    >
      <b>Generando predicción</b>
      <v-progress-linear
        color="primary"
        class="ml-3"
        :value="Math.round(progressStatus)"
      ></v-progress-linear>
      <p class="mb-0 ml-3">{{ Math.round(progressStatus) }}%</p>
    </div>
    <div
      class="mt-3"
      v-if="
        progressStatus === 100 &&
        Object.keys(singleFileUploadData).length > 0 &&
        uploadType === 'single'
      "
    >
      <v-row>
        <v-col cols="12" md="6" sm="12">
          <v-img
            v-if="uploadedImgSrc && progressStatus === 100"
            :src="uploadedImgSrc"
            class="img-size"
          />
        </v-col>
        <v-col cols="12" md="6" sm="12">
          <!--
          <v-img
            v-if="uploadedImgSrc && progressStatus === 100"
            :src="singleFileUploadData.image_url"
            class="ml-md-3 img-size"
          />
          -->
        </v-col>
      </v-row>

      <p class="mt-3 mb-0">
        <b>Tipo de Nube: </b>{{ singleFileUploadData.class }}
      </p>
      <p><b> Descripción: </b>{{ singleFileUploadData.description }}</p>
    </div>
    <v-btn
      color="primary"
      v-if="uploadType === 'multiple' && multipleFileUploadData.length === files.length"
      @click="generateCSV"
    >
      <v-img :src="require('@/assets/download.svg')" class="mr-2" />
      Descargar predicción
    </v-btn>
    <v-btn
      color="primary"
      outlined
      @click="reset"
      v-if="formSubmit && ((multipleFileUploadData.length === files.length) || Object.keys(singleFileUploadData).length > 0)"
      :class="{ 'ml-sm-3 mt-3 mt-sm-0': uploadType === 'multiple' }"
    >
      <v-img :src="require('@/assets/autorenew.svg')" class="mr-2" />
      Realizar nueva predicción
    </v-btn>
  </v-container>
</template>

<script>
import axios from "axios";
export default {
  name: "FileUpload",
  data: () => ({
    files: [],
    formSubmit: false,
    uploadedImgSrc: null,
    progressStatus: 0,
    singleFileUploadData: {},
    multipleFileUploadData: [],
    uploadType: "",
  }),
  components: {},
  watch:{
    files(newValue){
      if(newValue.length === 0) this.formSubmit = false
    }
  },
  methods: {
    reset() {
      this.formSubmit = false;
      this.progressStatus = 0;
      this.$refs.form.reset();
      this.uploadType = "";
      this.singleFileUploadData = {}
      this.multipleFileUploadData = []
    },
    async handleSubmit() {
      this.uploadedImgSrc = null;
      if (this.files.length === 1) {
        this.uploadType = "single";
        await this.singleFileUpload();
        this.uploadedImgSrc = URL.createObjectURL(this.files[0]);
        this.formSubmit = true;
      } else {
        this.uploadType = "multiple";
        await this.multipleFileUpload();
        this.formSubmit = true;
      }
    },
    singleFileUpload() {
      const options = {
        onUploadProgress: (progressEvent) => {
          const { loaded, total } = progressEvent;
          let precentage = Math.floor((loaded * 100) / total);
          if (precentage < 100) this.progressStatus = precentage;
          else this.progressStatus = 100;
        },
      };
      const form = new FormData();
      form.append("file", this.files[0]);
      axios
        .post(`${process.env.VUE_APP_BASE_URL}singleFile`, form, options)
        .then((res) => {
          if (res.data) {
            this.singleFileUploadData = res.data;
          }
        });
    },
    multipleFileUpload() {
      const totalProgress =  100 / this.files.length
      this.progressStatus = totalProgress
      const myUploadProgress = (index) => (progress) => {
        let percentage = Math.floor((progress.loaded * 100) / progress.total);
        if (percentage === 100 && index > 0) this.progressStatus += totalProgress;
        if (Math.round(this.progressStatus) >= 100) this.progressStatus = 100;
      };

      for (var i = 0; i < this.files.length; i++) {
        var config = {
          onUploadProgress: myUploadProgress(i),
        };
        const form = new FormData();
        form.append("file", this.files[i]);
        axios
          .post(`${process.env.VUE_APP_BASE_URL}multipleFiles`, form, config)
          .then((res) => {
            this.multipleFileUploadData.push({
              filename: res.data.file,
              class: res.data.class,
            });
          });
      }
    },
    generateCSV() {
      let csv = "File Name,Class\n";
      this.multipleFileUploadData.forEach((row) => {
        csv += row.filename + "," + row.class;
        csv += "\n";
      });
      console.log(csv);
      const anchor = document.createElement("a");
      anchor.acceptCharset = "UTF-8";
      anchor.href = "data:text/csv;charset=UTF-8," + encodeURIComponent(csv);
      anchor.target = "_blank";
      anchor.download = "results.csv";
      anchor.click();
    },
  },
};
</script>
<style scoped>
.disable-events {
  pointer-events: none;
  opacity: 0.5;
}

/deep/ .v-progress-linear {
  height: 8px !important;
  border-radius: 25px;
}


.img-size {
  width: 400px;
  height: 100%;
  margin: 0 auto;
}
</style>
