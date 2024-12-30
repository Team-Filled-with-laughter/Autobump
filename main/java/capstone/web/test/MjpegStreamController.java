package capstone.web.test;

import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.Map;

@Controller
public class MjpegStreamController {
    //private final String URL="182.209.79.20";
    private final String URL="localhost";
    // 환영 메시지와 비디오 스트리밍을 포함한 HTML 페이지 제공
    @GetMapping("/")
    public String welcomePage() {
        return "index";  // templates/index.html 페이지 반환
    }

    @GetMapping("/speed_feed")
    public SseEmitter getSpeedFeed() {
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);

        // 새로운 스레드에서 SSE 스트리밍 데이터를 읽어서 Spring 클라이언트로 전송
        new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new URL("http://" + URL + ":5001/speed_feed").openStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    // 데이터가 "data: {...}" 형태로 올 경우 처리
                    if (line.startsWith("data: ")) {
                        // JSON 데이터를 추출
                        String data = line.substring(6);  // "data: " 제거
                        // 클라이언트에 전송
                        emitter.send(data);
                    }
                }
            } catch (IOException e) {
                emitter.completeWithError(e); // 오류 처리
                System.err.println("SSE 스트리밍 오류: " + e.getMessage());
            } finally {
                emitter.complete();  // 스트리밍이 종료되면 emitter를 완료 상태로 처리
            }
        }).start();

        return emitter;
    }



    // MJPEG 스트리밍을 제공하는 경로
    @GetMapping("/video-feed")
    @ResponseBody
    public void streamVideo(HttpServletResponse response) throws IOException {
        URL url = new URL("http://" + URL + ":5001/video_feed");
        URLConnection connection = url.openConnection();

        // InputStream, OutputStream을 try-with-resources로 처리
        try (InputStream inputStream = connection.getInputStream();
             OutputStream responseOutputStream = response.getOutputStream()) {

            response.setContentType("multipart/x-mixed-replace; boundary=frame");
            response.setCharacterEncoding("UTF-8");

            byte[] buffer = new byte[1024];
            int bytesRead;

            // MJPEG 스트리밍 데이터를 읽어서 클라이언트로 전달
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                responseOutputStream.write(buffer, 0, bytesRead);
                responseOutputStream.flush();  // 즉시 전송
            }

        } catch (IOException e) {
            // 예외 처리 로직 (예: 로그 출력)
            System.err.println("스트리밍 오류: " + e.getMessage());
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "스트리밍 오류");
        }
    }

    @PostMapping("/update_threshold")
    public ResponseEntity<?> updateThreshold(@RequestBody ThresholdRequest thresholdRequest) {
        // 받은 임계값을 처리
        if (thresholdRequest.getThreshold() != null) {
            // 임계값 처리 로직 (예: 임계값을 DB에 저장하거나 다른 처리)
            return ResponseEntity.ok().body(Map.of("message", "임계값 업데이트 성공", "threshold", thresholdRequest.getThreshold()));
        } else {
            return ResponseEntity.badRequest().body(Map.of("error", "임계값을 찾을 수 없습니다."));
        }
    }

    @PostMapping("/update_actuator")
    public ResponseEntity<?> updateActuator(@RequestBody ActuatorRequest thresholdRequest) {
        // 받은 임계값을 처리
        if (thresholdRequest.getThreshold() != null) {
            return ResponseEntity.ok().body(Map.of("message", "엑추에이터 작동", "up", thresholdRequest.getThreshold()));
        } else {
            return ResponseEntity.badRequest().body(Map.of("error", "엑추에이터 작동오류"));
        }
    }

    public static class ActuatorRequest {
        private Integer up;

        // Getter and Setter
        public Integer getThreshold() {
            return up;
        }

        public void setThreshold(Integer threshold) {
            this.up = threshold;
        }
    }

    // 임계값을 받기 위한 DTO 클래스
    public static class ThresholdRequest {
        private Integer threshold;

        // Getter and Setter
        public Integer getThreshold() {
            return threshold;
        }

        public void setThreshold(Integer threshold) {
            this.threshold = threshold;
        }
    }
}
