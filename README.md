<!-- Improved compatibility of back to top link: See: https://github.com/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT LOGO AND TITLE -->
<p align="center">
  <a href="https://github.com/username/Project-Name">
  </a>

  <h2 align="center">DETEKSI TEMPAT DUDUK MENGGUNAKAN YOLOv11</h2>
  <p align="center">
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">Project ini</a></li>
    <li><a href="#start">Requirement System</a></li>
    <li><a href="#use">Runing Detection</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## <a name="about"></a>Project Ini

contoh project untuk input hari : Selasa, 11 April 2025, 20:54 WIB.

<img width="503" height="570" alt="image" src="https://github.com/user-attachments/assets/7108cf0d-0962-45a7-95c7-21f667d095e3" />

Di lingkungan coffee shop seperti JARAK COFFEE & EATERY, ketidaktersediaan informasi real-time tentang kursi kosong dapat menurunkan efisiensi layanan dan kepuasan pelanggan. Teknologi Computer Vision dan Deep Learning dapat dimanfaatkan untuk membuat sistem pemantauan otomatis yang mendeteksi kursi dan orang, lalu menentukan ketersediaan kursi berdasarkan hubungan spasial antara keduanya.

Tujuan Utama Project ini adalah:

- Mengembangkan sebuah model deteksi objek berbasis YOLO yang dapat mengenali objek orang dan kursi.
- Mengimplementasikan logika untuk menganalisis hubungan spasial antara bounding box "Orang" dan "Kursi" untuk menentukan status keterisian kursi.
- Membangun sebuah prototipe sistem yang dapat memproses umpan video secara real-time dan menampilkan hasil deteksi serta status kursi.

Keterbatasan Project ini:

- Performa menurun dalam kondisi pencahayaan rendah atau oklusi berat (orang menutupi kursi).
- MHanya menggunakan video rekaman, belum live stream.
- Hanya mendeteksi 2 kelas objek (kursi dan orang).

## <a name="start"></a>Requirement System

### Instalasai:

* Install the required tools:

  ```bash
  python 3.10 kebawah dikarenakan opencv untuk python 3.10 keatas belum optimal
  opencv-python==4.8.0
  ultralytics==8.0.0
  numpy==1.24.0
  torch==2.0.0

## <a name="use"></a>Runing Detection

Cara termudah untuk menjalankan program adalah membuka bash/terminal file dan melakukan perintah, list perintah untuk menjalankan program:

    python realtime_yolo_v11.py --source gbr_3.jpg  --model best.pt -> digunaakan untuk menjalankan input file berbentuk gambar (.png, .jpg, .jpeg, dst)
    python realtime_yolo_v11.py --source jarak_1.mp4  --model best.pt -> digunakan untuk menjalankan input file berbentuk video
    python realtime_yolo_v11.py --model best.pt --source 0 -> source adalah tempat webcam terrsebut disaambungakan contoh pada slot usb 0 -> digunakan untuk menjalankan kode menggunakan input kamera secara langsung 
    

If you face any issues, please refer to the `FAQ.md` or `HELP.md` for guidance.

[🔝 Back to Top](#readme-top-anchor)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
