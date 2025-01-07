import time
import threading

class Control:
    def __init__(self, process):
        self.stop_thread = False
        self.process = process
        self.output_thread = threading.Thread(target=self.read_output)
        
    def start_program(self):
        self.output_thread.start()
        print("Program started.")

    # 프로그램 종료 함수
    def stop_program(self):
        self.stop_thread = True  # 스레드를 종료하도록 플래그 설정
        if self.process.poll() is None:  # 프로그램이 실행 중이면 종료
            self.process.terminate()
            self.output_thread.join()  # 스레드가 종료될 때까지 기다림
        print("Program terminated.")

    """  Define functions for program control """        
    # 프로그램 출력을 실시간으로 읽어와 셀에 직접 출력하는 함수
    def read_output(self):
        while not self.stop_thread:
            if self.process.poll() is not None:
                # 프로세스 종료된 경우 루프 탈출
                break
            output = self.process.stdout.readline()
            if output:
                print(output.strip())
            else:
                # 출력이 없는 경우 잠시 대기 후 재시도
                time.sleep(0.1)
            
    # 사용자 입력을 프로그램에 전달하는 함수
    def send_input(self, user_input):
        if self.process.poll() is None:  # 프로그램이 실행 중인지 확인
            self.process.stdin.write(user_input + "\n")
            self.process.stdin.flush()  # 입력을 강제로 전달
            # print("Sent input:", user_input)  # 전송된 입력을 셀에 표시
        else:
            print("Program terminated.")
            
    def standby(self):
        # conect camera
        self.send_input("1")
        self.send_input("1")
        # save setting
        self.send_input("2")
        self.send_input("20")
        self.send_input("2") # CARD and SDRAM
        self.send_input("0") 
        time.sleep(4)

    def capture(self):
        self.send_input("10")

    def transfer_to_computer(self):
        self.send_input("1") # Select Item Object
        self.send_input("1") # Internal ID = {}
        self.send_input("1") # Select Data Object
        self.send_input("1") # Image
        self.send_input("6")

    def video_transfer_to_computer(self):
        self.send_input("1") # Select Item Object
        self.send_input("1") # Internal ID = {}
        self.send_input("1") # Select Data Object
        self.send_input("1") # Movie
        self.send_input("5") # GetVideoImageEx
        self.send_input("0") 
        self.send_input("0") 

    def video_capture_start(self):
        self.send_input("6")
        self.send_input("7")
        self.send_input("1")
        self.send_input("0")

    def video_capture_stop(self):
        self.send_input("6")
        self.send_input("7")
        self.send_input("0")
        self.send_input("0")

    def video_capture_stop_and_save(self):
        self.video_capture_stop()
        time.sleep(2)
        self.video_transfer_to_computer()
        time.sleep(3)

    def capture_and_save(self):
        self.capture()
        self.transfer_to_computer()
        time.sleep(2)

    def terminate(self):
        self.stop_thread = True  # 스레드를 종료하도록 플래그 설정
        if self.process.poll() is None:  # 프로그램이 아직 실행 중이라면
            # 프로그램 정상 종료 신호 전송 (상황에 따라 필요한 명령으로 변경)
            self.send_input("0")
            self.send_input("0")
            self.send_input("0")
            # 프로세스가 정상 종료되기를 잠시 대기
            try:
                self.process.wait(timeout=5)  # 5초 내에 종료 시도
            except:
                # 정해진 시간 내 종료 안 될 시 강제 종료
                self.process.terminate()

        # 출력 스레드를 종료하고 정리
        if self.output_thread.is_alive():
            self.output_thread.join()

        print("Program terminated.")