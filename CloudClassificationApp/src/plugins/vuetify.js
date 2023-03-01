import Vue from 'vue'
import Vuetify from 'vuetify'
import 'vuetify/dist/vuetify.min.css'

Vue.use(Vuetify)

const vuetify = new Vuetify({
    theme: {
      themes: {
        dark: {
          primary: '#042354',
        },
        light:{
            primary: '#042354',
        }
      },
    },
  })

export default vuetify