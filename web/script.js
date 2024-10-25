const videoFeed = document.getElementById('video-feed');
        const startBtn = document.getElementById('start-btn');

        startBtn.addEventListener('click', () => {
            videoFeed.src = '/video_feed';
        
});
gsap.from('.video-container',{duration: 1, y:'-100%',ease:'bounce'})
gsap.to('.video-container1',{duration: 10, y:'900%',ease:'bounce'})
gsap.to('.video-container2',{duration: 15, y:'900%',ease:'bounce'})
const button = document.querySelector('.submit');
button.addEventListener('click',()=>{
    document.querySelector(".submit").style.display = "none";
    gsap.to('.video-container',{duration: 1, y:'150%',ease:'bounce'})

});