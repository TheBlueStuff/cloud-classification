<template>
  <v-app id="app">
    <v-content>
    <v-app-bar
      extension-height="100%"
      :style="{ background: $vuetify.theme.themes.dark.primary }"
    >
      <v-toolbar-title
        ><v-img :src="require('@/assets/logo.svg')" />
      </v-toolbar-title>
    </v-app-bar>
    <div class="wrapper">
    <FileUpload/>
  </div>
    <v-footer color="transparent">
      <v-col
        cols="12"
      >
      <v-row class="d-flex align-center">
        <v-col cols="12" md="6" sm="12">
        <p class="mb-0 font-weight-medium black--text">Modelo de red neuronal convolucional para la clasificación de tipos de nubes en imágenes de webcam</p>
      </v-col>
      <v-col cols="12" md="6" sm="12" class="text-md-right">
        <p class="mb-0 font-weight-medium black--text">Mario Andonaire - 2022</p>
      </v-col>
      </v-row>
      </v-col>
    </v-footer>
    </v-content>
  </v-app>
</template>

<script>
import FileUpload from '@/components/FileUpload'
export default {
  props: {
  },
  created () {
  },
  name: "App",
  components: {FileUpload},
};
</script>

<style>
body{
  background: #ffffff;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #000000;
}
.v-application--wrap{
  min-height: auto !important;
}
.wrapper{
  min-height: 100vh !important;
}
</style>
